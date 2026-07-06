//! CircuitOpt Builder — the Tauri desktop shell around the browser circuit
//! editor. It owns the lifecycle of the Python `circuitopt` service:
//!
//!   * adopt a service the user already started on :8341 (never kill it), OR
//!   * discover + spawn one (config file → login-shell PATH), on a free port,
//!     wait for it to answer `/api/v1/health`, and kill it on quit.
//!
//! The negotiated API base is injected into the webview before it loads via
//! `window.__CIRCUITOPT_API_BASE__`, which `client.ts` reads at highest
//! priority. All the "how do we find/launch the backend" logic lives in
//! [`backend`]; this file is the Tauri glue + process bookkeeping.

mod backend;

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder};

/// PID of the backend *we* spawned, mirrored here so an async-signal-safe
/// handler can reap it without locking. `0` = nothing to kill (adopted service
/// or offline). Only `kill(2)` — which is async-signal-safe — is called from
/// the handler; no allocation, no mutex.
static BACKEND_PID: AtomicI32 = AtomicI32::new(0);

/// The service child *we* spawned, if any. `None` means we either adopted a
/// user-run service or found none — in both cases we must not kill anything on
/// exit. Guarded by a mutex so the exit handler and setup thread agree.
#[derive(Default)]
struct BackendProcess(Mutex<Option<Child>>);

/// Kill the spawned backend's whole process group (negative PID), if any. Safe
/// to call more than once. Used by both the normal `RunEvent::Exit` path and
/// the signal handler.
#[cfg(unix)]
fn kill_backend_group() {
    let pid = BACKEND_PID.swap(0, Ordering::SeqCst);
    if pid > 0 {
        // Negative → the whole process group we created via setpgid, so a
        // uvicorn reloader or any grandchildren go too.
        unsafe {
            libc::kill(-pid, libc::SIGTERM);
        }
    }
}

/// SIGTERM/SIGINT handler: reap the backend, then restore the default handler
/// and re-raise so the process still terminates with the expected disposition.
/// Everything here is async-signal-safe.
#[cfg(unix)]
extern "C" fn on_terminating_signal(sig: libc::c_int) {
    kill_backend_group();
    unsafe {
        libc::signal(sig, libc::SIG_DFL);
        libc::raise(sig);
    }
}

/// Install the SIGTERM/SIGINT handlers once, at startup.
#[cfg(unix)]
fn install_signal_handlers() {
    let handler = on_terminating_signal as *const () as libc::sighandler_t;
    unsafe {
        libc::signal(libc::SIGTERM, handler);
        libc::signal(libc::SIGINT, handler);
    }
}

/// Append a line to the app log file (best-effort; logging must never crash the
/// app). Also echoes to stderr so `tauri dev` shows it inline.
fn log_line(log_path: &PathBuf, msg: &str) {
    eprintln!("[circuitopt] {msg}");
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(log_path) {
        let _ = writeln!(f, "{msg}");
    }
}

/// Resolve, spawn and health-check the backend. Returns the API base URL the
/// webview should use, and — when we launched the process — the `Child` to kill
/// on exit. On total failure returns the default base with no child so the
/// frontend shows its offline banner.
fn start_backend(config_dir: PathBuf, log_path: PathBuf) -> (String, Option<Child>) {
    let default_base = format!("http://127.0.0.1:{}", backend::DEFAULT_PORT);

    // 1. Adopt an already-healthy service on the default port. We do NOT own it,
    //    so no child is returned and exit-cleanup leaves it alone.
    if backend::health_ok(backend::DEFAULT_PORT, Duration::from_millis(600)) {
        log_line(
            &log_path,
            &format!(
                "adopted existing backend on :{} (user-managed; will not be killed on quit)",
                backend::DEFAULT_PORT
            ),
        );
        return (default_base, None);
    }

    // 2. Resolve a launch command. Read the config file (writing a template on
    //    first run), then fall through to the login-shell PATH lookup.
    let cfg = load_or_init_config(&config_dir, &log_path);
    let resolved = match backend::resolve_command(&cfg) {
        Some(r) => r,
        None => {
            log_line(
                &log_path,
                "no backend found (config command null + circuit-opt not on login-shell PATH); \
                 starting in offline mode. Install with: pip install \"circuit-optimization[serve]\" \
                 or set \"command\" in backend.json.",
            );
            return (default_base, None);
        }
    };
    log_line(
        &log_path,
        &format!(
            "resolved backend via {:?}: {:?}",
            resolved.source, resolved.argv
        ),
    );

    // 3. Pick a free port (scan upward from the default) and spawn.
    let port = match backend::find_free_port(backend::DEFAULT_PORT, 100) {
        Some(p) => p,
        None => {
            log_line(&log_path, "no free port in [8341, 8441); offline mode");
            return (default_base, None);
        }
    };
    let argv = backend::with_serve_port(&resolved, port);
    let base = format!("http://127.0.0.1:{port}");
    log_line(&log_path, &format!("spawning: {argv:?}"));

    let child = match spawn_backend(&argv, &log_path) {
        Ok(c) => c,
        Err(e) => {
            log_line(&log_path, &format!("spawn failed: {e}; offline mode"));
            return (default_base, None);
        }
    };

    // 4. Wait (up to ~15s) for /api/v1/health to answer.
    if backend::wait_for_health(port, Duration::from_secs(15), Duration::from_millis(300)) {
        log_line(&log_path, &format!("backend healthy on :{port}"));
        // Record the PID for the signal handler (child.id() is the group leader
        // since we setpgid'd it into its own group).
        BACKEND_PID.store(child.id() as i32, Ordering::SeqCst);
        (base, Some(child))
    } else {
        log_line(
            &log_path,
            &format!("backend on :{port} never became healthy within 15s; offline mode"),
        );
        // Kill the stuck child so we don't leak it.
        let mut child = child;
        let _ = child.kill();
        let _ = child.wait();
        (default_base, None)
    }
}

/// Spawn `argv` with stdout+stderr redirected to the log file. `argv[0]` is the
/// executable; the rest are arguments. On Unix the child is placed in its own
/// process group (`setpgid(0, 0)`) so we can signal the whole group — reaping
/// any uvicorn-reloader grandchildren along with the server.
fn spawn_backend(argv: &[String], log_path: &PathBuf) -> std::io::Result<Child> {
    let out = File::options().create(true).append(true).open(log_path)?;
    let err = out.try_clone()?;
    let mut cmd = Command::new(&argv[0]);
    cmd.args(&argv[1..])
        .stdout(Stdio::from(out))
        .stderr(Stdio::from(err))
        .stdin(Stdio::null());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // SAFETY: setpgid is async-signal-safe and touches no shared state; it
        // just detaches the child into a new process group before exec.
        unsafe {
            cmd.pre_exec(|| {
                if libc::setpgid(0, 0) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }
    cmd.spawn()
}

/// Read `backend.json`, writing a documented template on first run if absent.
/// A malformed file is logged and treated as unconfigured (we don't clobber it).
fn load_or_init_config(config_dir: &PathBuf, log_path: &PathBuf) -> backend::BackendConfig {
    let _ = std::fs::create_dir_all(config_dir);
    let path = backend::config_path(config_dir);
    match std::fs::read_to_string(&path) {
        Ok(text) => match backend::parse_config(&text) {
            Ok(cfg) => cfg,
            Err(e) => {
                log_line(
                    log_path,
                    &format!(
                        "backend.json is malformed ({e}); ignoring it: {}",
                        path.display()
                    ),
                );
                backend::BackendConfig::default()
            }
        },
        Err(_) => {
            // First run (or unreadable): write the template.
            let template = backend::config_template();
            if let Ok(json) = serde_json::to_string_pretty(&template) {
                if std::fs::write(&path, json).is_ok() {
                    log_line(
                        log_path,
                        &format!("wrote backend.json template: {}", path.display()),
                    );
                }
            }
            template
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Reap the backend even on `kill <app>` / Ctrl-C, not just the GUI-quit path.
    #[cfg(unix)]
    install_signal_handlers();

    tauri::Builder::default()
        .manage(BackendProcess::default())
        .setup(|app| {
            // Tauri-managed dirs: config for backend.json, logs for the redirect.
            let config_dir = app
                .path()
                .app_config_dir()
                .unwrap_or_else(|_| PathBuf::from("."));
            let log_dir = app
                .path()
                .app_log_dir()
                .unwrap_or_else(|_| config_dir.clone());
            let _ = std::fs::create_dir_all(&log_dir);
            let log_path = log_dir.join("backend.log");
            log_line(&log_path, "── CircuitOpt Builder starting ──");

            // Resolve + launch the backend (blocks setup briefly; a local health
            // poll is fast, and doing it before window creation lets us inject
            // the final API base before any frontend code runs).
            let (api_base, child) = start_backend(config_dir, log_path.clone());
            *app.state::<BackendProcess>().0.lock().unwrap() = child;

            // Inject the negotiated base at the highest precedence tier (see
            // client.ts). Must run before the window's document loads, so we
            // create the window here with an initialization script rather than
            // declaring it in tauri.conf.json.
            let init_script = format!(
                "window.__CIRCUITOPT_API_BASE__ = {};",
                serde_json::to_string(&api_base).unwrap()
            );
            log_line(&log_path, &format!("API base for webview: {api_base}"));

            WebviewWindowBuilder::new(app, "main", WebviewUrl::default())
                .title("CircuitOpt Builder")
                .inner_size(1280.0, 860.0)
                .min_inner_size(900.0, 600.0)
                .initialization_script(&init_script)
                .build()?;

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building the CircuitOpt Builder application")
        .run(|app_handle, event| {
            // Normal GUI-quit path (Cmd+Q / last window closed). Kill ONLY a
            // backend we spawned; adopted/user-run services (child is None) are
            // left running. The signal handler covers `kill <app>` separately.
            if let RunEvent::Exit = event {
                let state = app_handle.state::<BackendProcess>();
                // Take the child out and drop the lock guard before touching it,
                // so the MutexGuard temporary doesn't outlive the borrow.
                let taken = state.0.lock().unwrap().take();
                if let Some(mut child) = taken {
                    #[cfg(unix)]
                    kill_backend_group(); // whole group, then reap the leader
                    let _ = child.kill();
                    let _ = child.wait();
                }
            }
        });
}
