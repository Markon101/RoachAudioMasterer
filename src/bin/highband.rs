//! Legacy backward-compatible binary entry point for `highband`.
//!
//! Executes `roach-audio-masterer` with all forwarded command-line arguments.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let current_exe = std::env::current_exe().unwrap_or_else(|_| "highband".into());
    let dir = current_exe.parent().unwrap_or(std::path::Path::new("."));
    let canonical_exe = dir.join("roach-audio-masterer");

    let exe_to_run = if canonical_exe.exists() {
        canonical_exe
    } else {
        std::path::PathBuf::from("roach-audio-masterer")
    };

    let status = std::process::Command::new(&exe_to_run).args(&args).status();

    match status {
        Ok(code) => {
            if let Some(c) = code.code() {
                std::process::exit(c);
            }
        }
        Err(e) => {
            eprintln!(
                "highband: failed to execute canonical binary {}: {e}",
                exe_to_run.display()
            );
            std::process::exit(1);
        }
    }
}
