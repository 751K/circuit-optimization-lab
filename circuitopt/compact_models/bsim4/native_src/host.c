#include <math.h>
#include <stddef.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/noisedef.h"
#include "ngspice/sperror.h"
#include "ngspice/tskdefs.h"
#include "bsim4v5def.h"
#include "bsim4v5ext.h"
#include "bsim4v5init.h"

#define CO_MAX_NODES 24
#define CO_STATES 256
#define CO_TERMINALS 4
#define CO_MAX_INTERNAL 2
#define CO_MAX_NOISE_SOURCES 16
#define CO_PI 3.14159265358979323846

struct MatrixFrame {
    double value[CO_MAX_NODES][CO_MAX_NODES][2];
};

typedef struct {
    int index;
    int node1;
    int node2;
    double density;
} CoNoiseSource;

typedef struct {
    BSIM4v5model model;
    BSIM4v5instance instance;
    CKTcircuit ckt;
    struct MatrixFrame matrix;
    double rhs[CO_MAX_NODES];
    double rhs_old[CO_MAX_NODES];
    double irhs[CO_MAX_NODES];
    double state0[CO_STATES];
    double state1[CO_STATES];
    CKTnode *nodes[CO_MAX_NODES];
    CoNoiseSource noise_sources[CO_MAX_NOISE_SOURCES];
    int noise_source_count;
    int setup_done;
    char error[256];
} CoBsim4;

double CONSTroot2 = 1.4142135623730950488;
double CONSTvt0 = 0.02586419;

static struct circ co_circ;
static TSKtask co_task;
struct circ *ft_curckt = &co_circ;

static void co_errorf(int level, const char *fmt, ...)
{
    (void)level;
    (void)fmt;
}

static IFfrontEnd co_frontend = {
    .IFerrorf = co_errorf,
};
IFfrontEnd *SPfrontEnd = &co_frontend;
FILE *slogp = NULL;
static __thread CoBsim4 *co_active_noise_device = NULL;

void *tmalloc(size_t size)
{
    return calloc(1, size);
}

void *trealloc(void *ptr, size_t size)
{
    return realloc(ptr, size);
}

void txfree(void *ptr)
{
    free(ptr);
}

bool cp_getvar(char *name, enum cp_types type, void *retval)
{
    (void)name;
    (void)type;
    (void)retval;
    return FALSE;
}

double *SMPmakeElt(SMPmatrix *matrix, int row, int col)
{
    struct MatrixFrame *dense = (struct MatrixFrame *)matrix;
    if (row < 0 || row >= CO_MAX_NODES || col < 0 || col >= CO_MAX_NODES)
        return NULL;
    return &dense->value[row][col][0];
}

int CKTmkVolt(CKTcircuit *ckt, CKTnode **result, IFuid basename, char *suffix)
{
    (void)basename;
    (void)suffix;
    CoBsim4 *device = (CoBsim4 *)((char *)ckt - offsetof(CoBsim4, ckt));
    int number = ++ckt->CKTmaxEqNum;
    if (number >= CO_MAX_NODES)
        return E_NOMEM;
    CKTnode *node = calloc(1, sizeof(*node));
    if (!node)
        return E_NOMEM;
    node->number = number;
    node->type = SP_VOLTAGE;
    device->nodes[number] = node;
    *result = node;
    return OK;
}

int CKTinst2Node(CKTcircuit *ckt, void *instance, int terminal,
                 CKTnode **node, IFuid *name)
{
    (void)ckt;
    (void)instance;
    (void)terminal;
    (void)node;
    (void)name;
    return E_NOTFOUND;
}

int CKTdltNNum(CKTcircuit *ckt, int number)
{
    (void)ckt;
    (void)number;
    return OK;
}

int NIintegrate(CKTcircuit *ckt, double *geq, double *ceq,
                double cap, int state)
{
    (void)cap;
    *geq = ckt->CKTag[0];
    ckt->CKTstate0[state + 1] =
        ckt->CKTag[0] * ckt->CKTstate0[state]
        + ckt->CKTag[1] * ckt->CKTstate1[state];
    *ceq = ckt->CKTstate0[state + 1];
    return OK;
}

void NevalSrc(double *noise, double *ln_noise, CKTcircuit *ckt,
              int type, int node1, int node2, double parameter)
{
    (void)node1;
    (void)node2;
    if (type == THERMNOISE)
        *noise = 4.0 * 1.380649e-23 * ckt->CKTtemp * parameter;
    else if (type == SHOTNOISE)
        *noise = 2.0 * 1.602176634e-19 * fabs(parameter);
    else if (type == N_GAIN)
        *noise = 1.0;
    else
        *noise = parameter;
    if (ln_noise)
        *ln_noise = log(fmax(*noise, 1.0e-38));
}

void CircuitOptBsim4NoiseSource(
    int index, int node1, int node2, double density)
{
    CoBsim4 *device = co_active_noise_device;
    if (!device || device->noise_source_count >= CO_MAX_NOISE_SOURCES)
        return;
    CoNoiseSource *source =
        &device->noise_sources[device->noise_source_count++];
    source->index = index;
    source->node1 = node1;
    source->node2 = node2;
    source->density = density;
}

double Nintegrate(double density, double ln_density, double ln_last, Ndata *data)
{
    (void)ln_density;
    (void)ln_last;
    (void)data;
    return density;
}

/*
 * The upstream checker writes model-derived values to bsim4v5.out. Native
 * CircuitOpt validates numeric cards and finite outputs without persisting
 * licensed model data, so the file-writing checker is intentionally omitted.
 */
int BSIM4v5checkModel(
    BSIM4v5model *model, BSIM4v5instance *instance, CKTcircuit *ckt)
{
    (void)model;
    (void)instance;
    (void)ckt;
    return 0;
}

static IFparm *co_find_param(IFparm *table, int count, const char *name)
{
    for (int i = 0; i < count; ++i) {
        if (strcasecmp(table[i].keyword, name) == 0)
            return &table[i];
    }
    return NULL;
}

static int co_set_param(IFparm *entry, double value, int model, CoBsim4 *device)
{
    IFvalue data;
    memset(&data, 0, sizeof(data));
    switch (entry->dataType & IF_VARTYPES) {
    case IF_INTEGER:
    case IF_FLAG:
        data.iValue = (int)llround(value);
        break;
    case IF_REAL:
        data.rValue = value;
        break;
    default:
        return E_BADPARM;
    }
    if (model)
        return BSIM4v5mParam(entry->id, &data, (GENmodel *)&device->model);
    return BSIM4v5param(entry->id, &data, (GENinstance *)&device->instance, NULL);
}

CoBsim4 *co_bsim4_create(int polarity, double temperature_k)
{
    CoBsim4 *device = calloc(1, sizeof(*device));
    if (!device)
        return NULL;

    co_task.jobs = NULL;
    co_circ.ci_curTask = &co_task;

    device->model.BSIM4v5modName = "circuitopt_bsim4";
    device->model.BSIM4v5instances = &device->instance;
    device->instance.BSIM4v5modPtr = &device->model;
    device->instance.BSIM4v5name = "m1";
    device->instance.BSIM4v5dNode = 1;
    device->instance.BSIM4v5gNodeExt = 2;
    device->instance.BSIM4v5sNode = 3;
    device->instance.BSIM4v5bNode = 4;

    device->ckt.CKTmatrix = (SMPmatrix *)&device->matrix;
    device->ckt.CKTrhs = device->rhs;
    device->ckt.CKTrhsOld = device->rhs_old;
    device->ckt.CKTirhs = device->irhs;
    device->ckt.CKTstate0 = device->state0;
    device->ckt.CKTstate1 = device->state1;
    device->ckt.CKTtemp = temperature_k;
    device->ckt.CKTnomTemp = 300.15;
    device->ckt.CKTmaxEqNum = 4;
    device->ckt.CKTabstol = 1.0e-12;
    device->ckt.CKTreltol = 1.0e-6;
    device->ckt.CKTvoltTol = 1.0e-6;
    device->ckt.CKTgmin = 1.0e-12;
    device->ckt.CKTbypass = 0;

    IFparm *type = co_find_param(
        BSIM4v5mPTable, BSIM4v5mPTSize, polarity > 0 ? "nmos" : "pmos");
    if (!type || co_set_param(type, 1.0, 1, device) != OK) {
        free(device);
        return NULL;
    }
    return device;
}

void co_bsim4_destroy(CoBsim4 *device)
{
    if (!device)
        return;
    for (int i = 0; i < CO_MAX_NODES; ++i)
        free(device->nodes[i]);
    free(device);
}

int co_bsim4_set_model(CoBsim4 *device, const char *name, double value)
{
    if (!device || device->setup_done)
        return E_NOCHANGE;
    IFparm *entry = co_find_param(BSIM4v5mPTable, BSIM4v5mPTSize, name);
    if (!entry)
        return E_BADPARM;
    return co_set_param(entry, value, 1, device);
}

int co_bsim4_set_instance(CoBsim4 *device, const char *name, double value)
{
    if (!device || device->setup_done)
        return E_NOCHANGE;
    IFparm *entry = co_find_param(BSIM4v5pTable, BSIM4v5pTSize, name);
    if (!entry)
        return E_BADPARM;
    return co_set_param(entry, value, 0, device);
}

int co_bsim4_setup(CoBsim4 *device)
{
    if (!device)
        return E_BADPARM;
    int states = 0;
    int status = BSIM4v5setup(
        (SMPmatrix *)&device->matrix, (GENmodel *)&device->model,
        &device->ckt, &states);
    if (status != OK)
        return status;
    device->ckt.CKTnumStates = states;
    status = BSIM4v5temp((GENmodel *)&device->model, &device->ckt);
    if (status == OK)
        device->setup_done = 1;
    return status;
}

int co_bsim4_node_count(const CoBsim4 *device)
{
    return device ? device->ckt.CKTmaxEqNum + 1 : 0;
}

static void co_clear(CoBsim4 *device)
{
    memset(&device->matrix, 0, sizeof(device->matrix));
    memset(device->rhs, 0, sizeof(device->rhs));
    memset(device->irhs, 0, sizeof(device->irhs));
}

static int co_solve_2x2(double a00, double a01, double a10, double a11,
                        double b0, double b1, double *x0, double *x1)
{
    double det = a00 * a11 - a01 * a10;
    if (fabs(det) < 1.0e-30)
        return E_PANIC;
    *x0 = (b0 * a11 - a01 * b1) / det;
    *x1 = (a00 * b1 - b0 * a10) / det;
    return OK;
}

typedef struct {
    double real;
    double imag;
} CoComplex;

static CoComplex co_complex_sub(CoComplex a, CoComplex b)
{
    CoComplex result = {a.real - b.real, a.imag - b.imag};
    return result;
}

static CoComplex co_complex_add(CoComplex a, CoComplex b)
{
    CoComplex result = {a.real + b.real, a.imag + b.imag};
    return result;
}

static CoComplex co_complex_mul(CoComplex a, CoComplex b)
{
    CoComplex result = {
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real,
    };
    return result;
}

static CoComplex co_complex_conjugate(CoComplex value)
{
    CoComplex result = {value.real, -value.imag};
    return result;
}

static CoComplex co_complex_div(CoComplex a, CoComplex b)
{
    double scale = b.real * b.real + b.imag * b.imag;
    CoComplex result = {
        (a.real * b.real + a.imag * b.imag) / scale,
        (a.imag * b.real - a.real * b.imag) / scale,
    };
    return result;
}

static CoComplex co_matrix_value(
    const CoBsim4 *device, int row, int col)
{
    CoComplex result = {
        device->matrix.value[row][col][0],
        device->matrix.value[row][col][1],
    };
    return result;
}

static int co_device_nodes(
    const CoBsim4 *device,
    int external[CO_TERMINALS],
    int internal[CO_MAX_INTERNAL])
{
    external[0] = device->instance.BSIM4v5dNode;
    external[1] = device->instance.BSIM4v5gNodeExt;
    external[2] = device->instance.BSIM4v5sNode;
    external[3] = device->instance.BSIM4v5bNode;
    int internal_count = 0;
    int candidates[] = {
        device->instance.BSIM4v5dNodePrime,
        device->instance.BSIM4v5gNodePrime,
        device->instance.BSIM4v5gNodeMid,
        device->instance.BSIM4v5sNodePrime,
        device->instance.BSIM4v5bNodePrime,
        device->instance.BSIM4v5dbNode,
        device->instance.BSIM4v5sbNode,
    };
    for (unsigned i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        int node = candidates[i];
        int is_external = 0;
        for (int j = 0; j < CO_TERMINALS; ++j)
            is_external |= node == external[j];
        int duplicate = 0;
        for (int j = 0; j < internal_count; ++j)
            duplicate |= node == internal[j];
        if (node > 0 && !is_external && !duplicate) {
            if (internal_count >= CO_MAX_INTERNAL)
                return -1;
            internal[internal_count++] = node;
        }
    }
    return internal_count;
}

int co_bsim4_dc(CoBsim4 *device, const double terminals[CO_TERMINALS],
                double currents[CO_TERMINALS],
                double conductance[CO_TERMINALS * CO_TERMINALS],
                double charges[CO_TERMINALS],
                double capacitance[CO_TERMINALS * CO_TERMINALS],
                double op[8])
{
    if (!device || !device->setup_done)
        return E_BADPARM;

    int external[CO_TERMINALS];
    int internal[CO_MAX_INTERNAL];
    int internal_count = co_device_nodes(device, external, internal);
    if (internal_count < 0)
        return E_UNSUPP;
    for (int i = 0; i < CO_TERMINALS; ++i)
        device->rhs_old[external[i]] = terminals[i];

    if (internal_count >= 1 && device->rhs_old[internal[0]] == 0.0)
        device->rhs_old[internal[0]] = terminals[0];
    if (internal_count >= 2 && device->rhs_old[internal[1]] == 0.0)
        device->rhs_old[internal[1]] = terminals[2];

    device->ckt.CKTmode = MODEDCOP | MODEINITFLOAT;
    for (int iteration = 0; iteration < 40; ++iteration) {
        co_clear(device);
        int status = BSIM4v5load((GENmodel *)&device->model, &device->ckt);
        if (status != OK)
            return status;
        if (internal_count == 0)
            break;

        double next[CO_MAX_INTERNAL] = {0.0, 0.0};
        if (internal_count == 1) {
            int row = internal[0];
            double b = device->rhs[row];
            for (int j = 0; j < CO_TERMINALS; ++j)
                b -= device->matrix.value[row][external[j]][0] * terminals[j];
            double a = device->matrix.value[row][row][0];
            if (fabs(a) < 1.0e-30)
                return E_PANIC;
            next[0] = b / a;
        } else {
            double b[CO_MAX_INTERNAL] = {
                device->rhs[internal[0]], device->rhs[internal[1]]};
            for (int row = 0; row < CO_MAX_INTERNAL; ++row)
                for (int j = 0; j < CO_TERMINALS; ++j)
                    b[row] -= device->matrix.value[internal[row]][external[j]][0]
                              * terminals[j];
            int status2 = co_solve_2x2(
                device->matrix.value[internal[0]][internal[0]][0],
                device->matrix.value[internal[0]][internal[1]][0],
                device->matrix.value[internal[1]][internal[0]][0],
                device->matrix.value[internal[1]][internal[1]][0],
                b[0], b[1], &next[0], &next[1]);
            if (status2 != OK)
                return status2;
        }
        double error = 0.0;
        for (int i = 0; i < internal_count; ++i) {
            error = fmax(error, fabs(next[i] - device->rhs_old[internal[i]]));
            device->rhs_old[internal[i]] = next[i];
        }
        if (error < 1.0e-12)
            break;
        if (iteration == 39)
            return E_PANIC;
    }

    co_clear(device);
    int status = BSIM4v5load((GENmodel *)&device->model, &device->ckt);
    if (status != OK)
        return status;

    for (int row = 0; row < CO_TERMINALS; ++row) {
        int r = external[row];
        double residual = -device->rhs[r];
        for (int col = 0; col < CO_TERMINALS; ++col)
            residual += device->matrix.value[r][external[col]][0] * terminals[col];
        for (int col = 0; col < internal_count; ++col)
            residual += device->matrix.value[r][internal[col]][0]
                        * device->rhs_old[internal[col]];
        currents[row] = residual;
    }

    for (int row = 0; row < CO_TERMINALS; ++row) {
        for (int col = 0; col < CO_TERMINALS; ++col) {
            double reduced = device->matrix.value[external[row]][external[col]][0];
            if (internal_count == 1) {
                int ii = internal[0];
                reduced -= device->matrix.value[external[row]][ii][0]
                           * device->matrix.value[ii][external[col]][0]
                           / device->matrix.value[ii][ii][0];
            } else if (internal_count == 2) {
                int i0 = internal[0], i1 = internal[1];
                double a00 = device->matrix.value[i0][i0][0];
                double a01 = device->matrix.value[i0][i1][0];
                double a10 = device->matrix.value[i1][i0][0];
                double a11 = device->matrix.value[i1][i1][0];
                double det = a00 * a11 - a01 * a10;
                double x0 = (
                    a11 * device->matrix.value[i0][external[col]][0]
                    - a01 * device->matrix.value[i1][external[col]][0]) / det;
                double x1 = (
                    -a10 * device->matrix.value[i0][external[col]][0]
                    + a00 * device->matrix.value[i1][external[col]][0]) / det;
                reduced -= device->matrix.value[external[row]][i0][0] * x0
                           + device->matrix.value[external[row]][i1][0] * x1;
            }
            conductance[row * CO_TERMINALS + col] = reduced;
        }
    }

    device->ckt.CKTmode = MODEDCOP | MODEINITSMSIG;
    co_clear(device);
    status = BSIM4v5load((GENmodel *)&device->model, &device->ckt);
    if (status != OK)
        return status;
    charges[0] = device->state0[device->instance.BSIM4v5qd];
    charges[1] = device->state0[device->instance.BSIM4v5qg];
    charges[2] = device->state0[device->instance.BSIM4v5qs];
    charges[3] = device->state0[device->instance.BSIM4v5qb];

    device->ckt.CKTomega = 1.0;
    co_clear(device);
    status = BSIM4v5acLoad((GENmodel *)&device->model, &device->ckt);
    if (status != OK)
        return status;
    for (int row = 0; row < CO_TERMINALS; ++row) {
        for (int col = 0; col < CO_TERMINALS; ++col) {
            CoComplex reduced =
                co_matrix_value(device, external[row], external[col]);
            if (internal_count == 1) {
                CoComplex left =
                    co_matrix_value(device, external[row], internal[0]);
                CoComplex center =
                    co_matrix_value(device, internal[0], internal[0]);
                CoComplex right =
                    co_matrix_value(device, internal[0], external[col]);
                reduced = co_complex_sub(
                    reduced, co_complex_mul(left, co_complex_div(right, center)));
            } else if (internal_count == 2) {
                CoComplex a00 = co_matrix_value(device, internal[0], internal[0]);
                CoComplex a01 = co_matrix_value(device, internal[0], internal[1]);
                CoComplex a10 = co_matrix_value(device, internal[1], internal[0]);
                CoComplex a11 = co_matrix_value(device, internal[1], internal[1]);
                CoComplex det = co_complex_sub(
                    co_complex_mul(a00, a11), co_complex_mul(a01, a10));
                CoComplex right0 =
                    co_matrix_value(device, internal[0], external[col]);
                CoComplex right1 =
                    co_matrix_value(device, internal[1], external[col]);
                CoComplex x0 = co_complex_div(
                    co_complex_sub(
                        co_complex_mul(a11, right0),
                        co_complex_mul(a01, right1)),
                    det);
                CoComplex x1 = co_complex_div(
                    co_complex_sub(
                        co_complex_mul(a00, right1),
                        co_complex_mul(a10, right0)),
                    det);
                CoComplex left0 =
                    co_matrix_value(device, external[row], internal[0]);
                CoComplex left1 =
                    co_matrix_value(device, external[row], internal[1]);
                reduced = co_complex_sub(
                    reduced,
                    co_complex_add(
                        co_complex_mul(left0, x0),
                        co_complex_mul(left1, x1)));
            }
            capacitance[row * CO_TERMINALS + col] = reduced.imag;
        }
    }

    op[0] = device->instance.BSIM4v5cd;
    op[1] = device->instance.BSIM4v5gm;
    op[2] = device->instance.BSIM4v5gds;
    op[3] = device->instance.BSIM4v5gmbs;
    op[4] = device->instance.BSIM4v5von;
    op[5] = device->instance.BSIM4v5vdsat;
    op[6] = device->instance.BSIM4v5ueff;
    op[7] = (double)internal_count;
    return OK;
}

int co_bsim4_noise(
    CoBsim4 *device, double frequency_hz,
    double total_real[CO_TERMINALS * CO_TERMINALS],
    double total_imag[CO_TERMINALS * CO_TERMINALS],
    double flicker_real[CO_TERMINALS * CO_TERMINALS],
    double flicker_imag[CO_TERMINALS * CO_TERMINALS])
{
    if (!device || !device->setup_done || !(frequency_hz > 0.0))
        return E_BADPARM;

    int external[CO_TERMINALS];
    int internal[CO_MAX_INTERNAL];
    int internal_count = co_device_nodes(device, external, internal);
    if (internal_count < 0)
        return E_UNSUPP;

    device->ckt.CKTomega = 2.0 * CO_PI * frequency_hz;
    co_clear(device);
    int status = BSIM4v5acLoad((GENmodel *)&device->model, &device->ckt);
    if (status != OK)
        return status;

    NOISEAN job;
    Ndata data;
    memset(&job, 0, sizeof(job));
    memset(&data, 0, sizeof(data));
    job.NstartFreq = frequency_hz;
    data.freq = frequency_hz;
    data.GainSqInv = 1.0;
    device->noise_source_count = 0;
    JOB *previous_job = device->ckt.CKTcurJob;
    device->ckt.CKTcurJob = (JOB *)&job;
    co_active_noise_device = device;
    double total_density = 0.0;
    status = BSIM4v5noise(
        N_DENS, N_CALC, (GENmodel *)&device->model,
        &device->ckt, &data, &total_density);
    co_active_noise_device = NULL;
    device->ckt.CKTcurJob = previous_job;
    if (status != OK)
        return status;

    CoComplex total[CO_TERMINALS][CO_TERMINALS];
    CoComplex flicker[CO_TERMINALS][CO_TERMINALS];
    memset(total, 0, sizeof(total));
    memset(flicker, 0, sizeof(flicker));

    for (int source_index = 0;
         source_index < device->noise_source_count;
         ++source_index) {
        CoNoiseSource *source = &device->noise_sources[source_index];
        if (!isfinite(source->density) || source->density < 0.0)
            return E_PARMVAL;
        if (source->density == 0.0)
            continue;

        CoComplex external_incidence[CO_TERMINALS];
        CoComplex internal_incidence[CO_MAX_INTERNAL];
        memset(external_incidence, 0, sizeof(external_incidence));
        memset(internal_incidence, 0, sizeof(internal_incidence));
        for (int terminal = 0; terminal < CO_TERMINALS; ++terminal) {
            if (source->node1 == external[terminal])
                external_incidence[terminal].real += 1.0;
            if (source->node2 == external[terminal])
                external_incidence[terminal].real -= 1.0;
        }
        for (int index = 0; index < internal_count; ++index) {
            if (source->node1 == internal[index])
                internal_incidence[index].real += 1.0;
            if (source->node2 == internal[index])
                internal_incidence[index].real -= 1.0;
        }

        CoComplex internal_voltage[CO_MAX_INTERNAL];
        memset(internal_voltage, 0, sizeof(internal_voltage));
        if (internal_count == 1) {
            internal_voltage[0] = co_complex_div(
                internal_incidence[0],
                co_matrix_value(device, internal[0], internal[0]));
        } else if (internal_count == 2) {
            CoComplex a00 = co_matrix_value(device, internal[0], internal[0]);
            CoComplex a01 = co_matrix_value(device, internal[0], internal[1]);
            CoComplex a10 = co_matrix_value(device, internal[1], internal[0]);
            CoComplex a11 = co_matrix_value(device, internal[1], internal[1]);
            CoComplex determinant = co_complex_sub(
                co_complex_mul(a00, a11), co_complex_mul(a01, a10));
            internal_voltage[0] = co_complex_div(
                co_complex_sub(
                    co_complex_mul(a11, internal_incidence[0]),
                    co_complex_mul(a01, internal_incidence[1])),
                determinant);
            internal_voltage[1] = co_complex_div(
                co_complex_sub(
                    co_complex_mul(a00, internal_incidence[1]),
                    co_complex_mul(a10, internal_incidence[0])),
                determinant);
        }

        CoComplex reduced[CO_TERMINALS];
        for (int terminal = 0; terminal < CO_TERMINALS; ++terminal) {
            reduced[terminal] = external_incidence[terminal];
            for (int index = 0; index < internal_count; ++index) {
                reduced[terminal] = co_complex_sub(
                    reduced[terminal],
                    co_complex_mul(
                        co_matrix_value(device, external[terminal], internal[index]),
                        internal_voltage[index]));
            }
        }
        for (int row = 0; row < CO_TERMINALS; ++row) {
            for (int col = 0; col < CO_TERMINALS; ++col) {
                CoComplex contribution = co_complex_mul(
                    reduced[row], co_complex_conjugate(reduced[col]));
                contribution.real *= source->density;
                contribution.imag *= source->density;
                total[row][col] = co_complex_add(total[row][col], contribution);
                if (source->index == BSIM4v5FLNOIZ) {
                    flicker[row][col] =
                        co_complex_add(flicker[row][col], contribution);
                }
            }
        }
    }

    for (int row = 0; row < CO_TERMINALS; ++row) {
        for (int col = 0; col < CO_TERMINALS; ++col) {
            int offset = row * CO_TERMINALS + col;
            total_real[offset] = total[row][col].real;
            total_imag[offset] = total[row][col].imag;
            flicker_real[offset] = flicker[row][col].real;
            flicker_imag[offset] = flicker[row][col].imag;
        }
    }
    return OK;
}
