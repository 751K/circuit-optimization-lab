//! Backend discovery, command resolution and port negotiation.
//!
//! This is the one place that knows *how* to find and launch the Python
//! service. The v1 story is "the user `pip install`s the package; the .app is a
//! thin shell that discovers and drives the process". If we later ship a frozen
//! PyInstaller sidecar, only [`resolve_command`] (and its callers in `lib.rs`)
//! change — the port negotiation and health polling stay put.
//!
//! Resolution order (see [`resolve_command`]):
//!   1. `backend.json` config file — `{"command": ["/path/python", "-m", ...]}`
//!   2. login-shell lookup of `circuit-opt` on PATH
//!   3. give up → the frontend falls back to its offline banner
//!
//! Everything expensive (spawning a shell, opening sockets) is factored so the
//! decision logic stays pure and unit-testable; see the `tests` module.

use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Default port the service binds when the user runs it themselves. We probe it
/// first (adopt-not-spawn) and, when we do spawn, start scanning here upward.
pub const DEFAULT_PORT: u16 = 8341;

/// On-disk shape of `backend.json`. `command` is the full argv of an
/// executable that accepts `serve --port <p>` (a `circuit-opt` entry point or a
/// `python -m circuitopt.service`-style invocation). `null` means "not
/// configured — fall through to shell lookup".
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BackendConfig {
    #[serde(default)]
    pub command: Option<Vec<String>>,
    /// A human hint we write into the template on first run; ignored on read.
    #[serde(rename = "_hint", default, skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

/// The template written to `backend.json` the first time the app runs and the
/// file is absent. Documents the field so a user can point us at their
/// interpreter without reading the docs.
pub fn config_template() -> BackendConfig {
    BackendConfig {
        command: None,
        hint: Some(
            "Set \"command\" to the argv that starts the circuitopt service, e.g. \
             [\"/path/to/python\", \"-m\", \"circuitopt.service\"] or \
             [\"/path/to/circuit-opt\", \"serve\"]. The app appends \"--port <n>\". \
             Leave null to auto-discover circuit-opt on your login-shell PATH."
                .to_string(),
        ),
    }
}

/// Where a resolved command came from — surfaced in logs so a puzzled user can
/// tell whether their config file or the PATH lookup won.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandSource {
    Config,
    LoginShell,
}

/// A resolved way to launch the backend: `argv[0]` is the executable, the rest
/// are leading arguments; the caller appends `serve --port <p>` (or just
/// `--port <p>` when the argv already ends in a `serve`-like token — see
/// [`with_serve_port`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedCommand {
    pub argv: Vec<String>,
    pub source: CommandSource,
}

/// Parse `backend.json` bytes into a config, tolerating an empty/whitespace
/// file (treated as "unconfigured"). A malformed file is an error the caller
/// logs and then falls through past config.
pub fn parse_config(bytes: &str) -> Result<BackendConfig, serde_json::Error> {
    let trimmed = bytes.trim();
    if trimmed.is_empty() {
        return Ok(BackendConfig::default());
    }
    serde_json::from_str(trimmed)
}

/// Pure step 1: turn a (maybe-present) config into a command, if it names one.
/// A config with `command: null` or an empty argv yields `None` so the caller
/// moves on to the shell lookup.
pub fn command_from_config(cfg: &BackendConfig) -> Option<ResolvedCommand> {
    match &cfg.command {
        Some(argv) if !argv.is_empty() && !argv[0].trim().is_empty() => Some(ResolvedCommand {
            argv: argv.clone(),
            source: CommandSource::Config,
        }),
        _ => None,
    }
}

/// Build the final argv to spawn: append `serve` then `--port <port>`, unless
/// the resolved argv already ends with a `serve` token (e.g. a config that
/// wrote `["circuit-opt", "serve"]`), in which case we only add the port. A
/// `python -m circuitopt.service` invocation has no `serve` token and its
/// `__main__` takes `--port` directly, so appending `serve` would be wrong for
/// it — we special-case that too.
pub fn with_serve_port(cmd: &ResolvedCommand, port: u16) -> Vec<String> {
    let mut argv = cmd.argv.clone();
    let ends_with_serve = argv.last().map(|s| s == "serve").unwrap_or(false);
    // `-m circuitopt.service` (the module entry point) parses `--port` itself
    // and must NOT get a `serve` subcommand.
    let is_module_service = argv
        .windows(2)
        .any(|w| w[0] == "-m" && w[1] == "circuitopt.service");
    if !ends_with_serve && !is_module_service {
        argv.push("serve".to_string());
    }
    argv.push("--port".to_string());
    argv.push(port.to_string());
    argv
}

/// Step 2: resolve `circuit-opt` through a **login** shell.
///
/// This matters: a GUI app launched from Finder inherits a bare PATH
/// (`/usr/bin:/bin:/usr/sbin:/sbin`) — conda/homebrew/pyenv shims are invisible.
/// `zsh -lc` sources the user's profile so `command -v circuit-opt` sees what a
/// terminal would. Returns the absolute path on success.
pub fn resolve_via_login_shell() -> Option<ResolvedCommand> {
    login_shell_which("circuit-opt").map(|path| ResolvedCommand {
        argv: vec![path],
        source: CommandSource::LoginShell,
    })
}

/// Run `<shell> -lc "command -v <bin>"` and return the first non-empty line.
/// Factored out so the resolution logic can be exercised without a real binary
/// on PATH.
fn login_shell_which(bin: &str) -> Option<String> {
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/zsh".to_string());
    let output = Command::new(shell)
        .arg("-lc")
        .arg(format!("command -v {bin}"))
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let path = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())?
        .to_string();
    if path.is_empty() {
        None
    } else {
        Some(path)
    }
}

/// Full resolution: config first (from an already-read config), then shell.
/// The config is passed in (not read here) so this stays pure and testable; the
/// caller in `lib.rs` handles the filesystem.
pub fn resolve_command(cfg: &BackendConfig) -> Option<ResolvedCommand> {
    command_from_config(cfg).or_else(resolve_via_login_shell)
}

/// Find a free loopback port at/after `start`, scanning upward. Returns `None`
/// if the whole window is taken (pathological). We bind-and-drop to test
/// availability, which races with a subsequent spawn but is fine for a
/// single-user desktop app.
pub fn find_free_port(start: u16, window: u16) -> Option<u16> {
    (start..start.saturating_add(window))
        .find(|&p| TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, p))).is_ok())
}

/// Hit `GET /api/v1/health` on `127.0.0.1:port` and return true iff it answers
/// `200` with a body containing `"status"` and `"ok"`. A hand-rolled tiny HTTP
/// GET keeps the crate dependency-free (no reqwest/hyper) — we only ever call
/// loopback.
pub fn health_ok(port: u16, timeout: Duration) -> bool {
    fetch_health(port, timeout)
        .map(|body| body.contains("\"status\"") && body.contains("ok"))
        .unwrap_or(false)
}

fn fetch_health(port: u16, timeout: Duration) -> Option<String> {
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    let mut stream = TcpStream::connect_timeout(&addr, timeout).ok()?;
    stream.set_read_timeout(Some(timeout)).ok()?;
    stream.set_write_timeout(Some(timeout)).ok()?;
    let req = "GET /api/v1/health HTTP/1.1\r\n\
               Host: 127.0.0.1\r\n\
               Connection: close\r\n\r\n";
    stream.write_all(req.as_bytes()).ok()?;
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).ok()?;
    let text = String::from_utf8_lossy(&buf);
    // Require a 200 status line; anything else (404, connection reset) fails.
    let (headers, body) = text.split_once("\r\n\r\n")?;
    if !headers.starts_with("HTTP/1.1 200") && !headers.starts_with("HTTP/1.0 200") {
        return None;
    }
    Some(body.to_string())
}

/// Poll [`health_ok`] until it passes or `deadline` elapses, sleeping `interval`
/// between tries. Returns true on success.
pub fn wait_for_health(port: u16, deadline: Duration, interval: Duration) -> bool {
    let start = std::time::Instant::now();
    loop {
        if health_ok(port, Duration::from_millis(500)) {
            return true;
        }
        if start.elapsed() >= deadline {
            return false;
        }
        std::thread::sleep(interval);
    }
}

/// Resolve the `backend.json` path under the app config dir, creating the dir.
/// Split out so `lib.rs` can pass the Tauri-provided config dir in.
pub fn config_path(config_dir: &Path) -> PathBuf {
    config_dir.join("backend.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_config_empty_is_unconfigured() {
        let cfg = parse_config("   \n ").unwrap();
        assert!(cfg.command.is_none());
    }

    #[test]
    fn parse_config_null_command() {
        let cfg = parse_config(r#"{"command": null, "_hint": "x"}"#).unwrap();
        assert!(cfg.command.is_none());
        assert!(command_from_config(&cfg).is_none());
    }

    #[test]
    fn parse_config_argv() {
        let cfg =
            parse_config(r#"{"command": ["/py/bin/python", "-m", "circuitopt.service"]}"#).unwrap();
        let resolved = command_from_config(&cfg).expect("config command");
        assert_eq!(resolved.source, CommandSource::Config);
        assert_eq!(
            resolved.argv,
            vec!["/py/bin/python", "-m", "circuitopt.service"]
        );
    }

    #[test]
    fn parse_config_empty_argv_is_none() {
        let cfg = parse_config(r#"{"command": []}"#).unwrap();
        assert!(command_from_config(&cfg).is_none());
    }

    #[test]
    fn parse_config_blank_exe_is_none() {
        let cfg = parse_config(r#"{"command": ["  "]}"#).unwrap();
        assert!(command_from_config(&cfg).is_none());
    }

    #[test]
    fn malformed_config_errs() {
        assert!(parse_config("{not json").is_err());
    }

    #[test]
    fn with_serve_port_appends_serve_for_plain_exe() {
        let cmd = ResolvedCommand {
            argv: vec!["/usr/local/bin/circuit-opt".to_string()],
            source: CommandSource::LoginShell,
        };
        assert_eq!(
            with_serve_port(&cmd, 8342),
            vec!["/usr/local/bin/circuit-opt", "serve", "--port", "8342"]
        );
    }

    #[test]
    fn with_serve_port_skips_serve_when_already_present() {
        let cmd = ResolvedCommand {
            argv: vec!["circuit-opt".to_string(), "serve".to_string()],
            source: CommandSource::Config,
        };
        assert_eq!(
            with_serve_port(&cmd, 9000),
            vec!["circuit-opt", "serve", "--port", "9000"]
        );
    }

    #[test]
    fn with_serve_port_module_service_gets_no_serve_token() {
        let cmd = ResolvedCommand {
            argv: vec![
                "/py/bin/python".to_string(),
                "-m".to_string(),
                "circuitopt.service".to_string(),
            ],
            source: CommandSource::Config,
        };
        // `python -m circuitopt.service --port N` — NOT `... service serve --port N`.
        assert_eq!(
            with_serve_port(&cmd, 8500),
            vec![
                "/py/bin/python",
                "-m",
                "circuitopt.service",
                "--port",
                "8500"
            ]
        );
    }

    #[test]
    fn config_beats_shell_in_resolve() {
        // With a config command present, resolve_command must not invoke the
        // shell lookup at all — the config wins.
        let cfg =
            parse_config(r#"{"command": ["/py/bin/python", "-m", "circuitopt.service"]}"#).unwrap();
        let resolved = resolve_command(&cfg).expect("resolved");
        assert_eq!(resolved.source, CommandSource::Config);
    }

    #[test]
    fn find_free_port_returns_something() {
        // On any dev/CI box some port at/after a high base is free.
        let p = find_free_port(49200, 200).expect("a free port exists");
        assert!((49200..49400).contains(&p));
    }

    #[test]
    fn find_free_port_detects_taken_port() {
        // Bind a listener, then confirm the scan skips it.
        let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0))).unwrap();
        let taken = listener.local_addr().unwrap().port();
        // A one-wide window over the taken port yields None.
        assert_eq!(find_free_port(taken, 1), None);
    }

    #[test]
    fn health_ok_false_when_nothing_listening() {
        // Pick a port nothing should be on.
        assert!(!health_ok(59999, Duration::from_millis(100)));
    }

    #[test]
    fn config_path_joins() {
        let p = config_path(Path::new("/a/b"));
        assert_eq!(p, PathBuf::from("/a/b/backend.json"));
    }

    #[test]
    fn config_template_has_null_command_and_hint() {
        let t = config_template();
        assert!(t.command.is_none());
        assert!(t.hint.is_some());
        // Round-trips to JSON with both fields.
        let s = serde_json::to_string(&t).unwrap();
        assert!(s.contains("\"command\":null"));
        assert!(s.contains("_hint"));
    }
}
