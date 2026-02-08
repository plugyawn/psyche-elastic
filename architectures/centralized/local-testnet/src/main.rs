use anyhow::{Context, Result, bail};
use clap::{ArgAction, Parser};
use rand::seq::SliceRandom;
use serde::Deserialize;
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use time::OffsetDateTime;
use time::macros::format_description;

#[derive(Parser, Debug)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[allow(clippy::large_enum_variant)] // it's only used for generating the docs correctly.
#[derive(Parser, Debug)]
enum Commands {
    /// Starts the local-testnet running each part of the system in a separate terminal pane.
    Start {
        #[command(flatten)]
        start_args: StartArgs,
    },
    // Prints the help, optionally as markdown. Used for docs generation.
    #[clap(hide = true)]
    PrintAllHelp {
        #[arg(long, required = true)]
        markdown: bool,
    },
}

#[derive(Parser, Debug, Clone)]
struct StartArgs {
    /// Number of clients to start
    #[clap(long, value_parser = validate_num_clients)]
    num_clients: usize,

    /// File path to the configuration that the coordinator will need to start.
    #[clap(long,value_parser = validate_config_path)]
    config_path: PathBuf,

    /// If provided, write DisTrO data to disk in this path.
    #[clap(long)]
    write_distro_data: Option<PathBuf>,

    /// Port where the server for this testnet will be listen it to (this is the one that clients must use when connecting).
    #[clap(long, default_value_t = 20000)]
    server_port: u16,

    /// Enables a terminal-based graphical interface for monitoring analytics.
    #[clap(
        long,
        action = ArgAction::Set,
        default_value_t = true,
        default_missing_value = "true",
        num_args = 0..=1,
        require_equals = false,
        env
    )]
    tui: bool,

    /// Run without tmux (spawns server/clients as subprocesses in this terminal).
    #[clap(long, default_value_t = false)]
    headless: bool,

    /// If set, automatically stop a headless testnet after N seconds.
    #[clap(long)]
    headless_exit_after_secs: Option<u64>,

    /// Skip running `nvtop` in the tmux monitor pane (useful on non-Linux).
    #[clap(long, default_value_t = false)]
    no_nvtop: bool,

    /// Kill N clients randomly every <RANDOM_KILL_INTERVAL> seconds
    #[clap(long)]
    random_kill_num: Option<usize>,

    /// Which clients we're allowed to kill randomly
    #[clap(long, value_delimiter = ',', default_values_t = &[])]
    allowed_to_kill: Vec<usize>,

    #[clap(long, default_value_t = 120)]
    /// Kill <RANDOM_KILL_NUM> clients randomly every N seconds
    random_kill_interval: u64,

    /// Sets the level of the logging for more granular information
    #[clap(long, default_value = "warn,psyche=debug")]
    log: String,

    /// Device(s) to use for clients (passed through to `psyche-centralized-client --device`).
    #[clap(long, default_value = "auto")]
    client_device: String,

    /// MatFormer tier(s) to pass to clients (comma-separated).
    ///
    /// If multiple tiers are provided, they are assigned to clients in a repeating cycle.
    ///
    /// Tier `0` = largest, higher = smaller.
    #[clap(long, value_delimiter = ',')]
    client_matformer_tiers: Vec<u8>,

    /// MatFormer helper fraction(s) for each client (comma-separated).
    ///
    /// When > 0, clients will also train a random sample of suffix neurons.
    /// Values should be in range 0.0-1.0. Assigned in cycle like tiers.
    #[clap(long, value_delimiter = ',')]
    client_matformer_helper_fractions: Vec<f32>,

    /// MatFormer helper rotation interval(s) for each client (comma-separated).
    ///
    /// How many rounds to keep helper indices fixed before rotating.
    /// Assigned in cycle like tiers.
    #[clap(long, value_delimiter = ',')]
    client_matformer_helper_rotation_intervals: Vec<u64>,

    /// MatFormer distillation beta_max (0 = disabled). Set > 0 to enable distillation.
    #[clap(long, default_value_t = 0.0)]
    matformer_distillation_beta_max: f64,

    /// MatFormer distillation warmup steps.
    #[clap(long, default_value_t = 100)]
    matformer_distillation_warmup_steps: u32,

    /// MatFormer distillation start step.
    #[clap(long, default_value_t = 0)]
    matformer_distillation_start_step: u32,

    /// MatFormer distillation top-k logits.
    #[clap(long, default_value_t = 32)]
    matformer_distillation_top_k: u16,

    /// MatFormer distillation temperature.
    #[clap(long, default_value_t = 2.0)]
    matformer_distillation_temperature: f32,

    /// What discovery mode to use for spawned clients (`local` or `n0`).
    #[clap(long, default_value = "local")]
    client_iroh_discovery: String,

    /// What relay kind to use for spawned clients (`disabled`, `psyche`, or `n0`).
    #[clap(long, default_value = "disabled")]
    client_iroh_relay: String,

    /// HF repo where the first client could get the model and the configuration to use.
    #[clap(long)]
    first_client_checkpoint: Option<String>,

    // HF token for all the clients to fetch the model at the beggining of the run.
    #[clap(long)]
    hf_token: Option<String>,

    #[clap(long, default_value_t = false)]
    write_log: bool,

    #[clap(long, env)]
    wandb_project: Option<String>,

    #[clap(long, env)]
    wandb_group: Option<String>,

    #[clap(long, env)]
    wandb_entity: Option<String>,

    /// Enable WandB step-level logging (every training step, not just rounds)
    #[clap(long, env, default_value_t = false)]
    wandb_step_logging: bool,

    /// Enable WandB system metrics logging (GPU usage, memory, temperature)
    #[clap(long, env, default_value_t = false)]
    wandb_system_metrics: bool,

    /// WandB system metrics logging interval in seconds
    #[clap(long, env, default_value_t = 10)]
    wandb_system_metrics_interval_secs: u64,

    #[clap(long, env)]
    optim_stats: Option<u32>,

    #[clap(long, env)]
    eval_tasks: Option<String>,

    /// Directory to save model checkpoints locally (passed to all clients).
    #[clap(long)]
    client_checkpoint_dir: Option<PathBuf>,

    /// Number of checkpoint steps to keep (default: 3).
    #[clap(long, default_value_t = 3)]
    client_keep_steps: u32,

    /// Inject network latency for stress testing. Format: "base_ms" or "base_ms-jitter_ms"
    /// Example: "50" = 50ms fixed, "50-20" = 50ms +/- 20ms uniform jitter
    #[clap(long, env)]
    fault_latency_ms: Option<String>,

    /// Packet loss probability for stress testing (0.0 to 1.0)
    /// Example: 0.1 = 10% packet loss
    #[clap(long, env)]
    fault_packet_loss: Option<f64>,

    /// Bandwidth limit in bytes per second for stress testing
    #[clap(long, env)]
    fault_bandwidth_limit: Option<u64>,

    /// Random seed for fault injection (for reproducibility)
    #[clap(long, env)]
    fault_seed: Option<u64>,
}

fn validate_num_clients(s: &str) -> Result<usize> {
    let n: usize = s
        .parse()
        .context("NUM_CLIENTS must be a positive integer")?;
    if n > 0 {
        Ok(n)
    } else {
        bail!("NUM_CLIENTS must be a positive integer")
    }
}

fn validate_config_path(s: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(s);
    if path.exists() {
        Ok(path)
    } else {
        Err(format!("Config path {s} does not exist"))
    }
}

#[derive(Deserialize)]
struct TomlWithRunId {
    run_id: String,
}

fn extract_run_id(state_path: &PathBuf) -> Result<String> {
    let toml: TomlWithRunId = toml::from_str(&std::fs::read_to_string(state_path)?)?;
    Ok(toml.run_id)
}

struct ChildProcesses {
    children: Vec<Child>,
}

impl Drop for ChildProcesses {
    fn drop(&mut self) {
        for child in &mut self.children {
            let _ = child.kill();
        }
        for child in &mut self.children {
            let _ = child.wait();
        }
    }
}

fn main() -> Result<()> {
    #[cfg(feature = "python")]
    psyche_python_extension_impl::init_embedded_python()?;

    let args = Args::parse();
    let command = args.command;

    match command {
        Commands::Start { mut start_args } => {
            if let Some(n_kill) = start_args.random_kill_num {
                if n_kill > start_args.num_clients {
                    bail!(
                        "You requested to kill {n_kill} clients randomly, but you only have {} clients.",
                        start_args.num_clients
                    );
                }
            }
            let state_path = start_args.config_path.join("state.toml");
            let data_path = start_args.config_path.join("data.toml");

            println!("{start_args:?}");

            if start_args.headless_exit_after_secs.is_some() && !start_args.headless {
                eprintln!("Note: --headless-exit-after-secs implies --headless");
                start_args.headless = true;
            }

            // Pre-build packages in release mode for libtorch compatibility
            Command::new("cargo")
                .args(["build", "--release", "-p", "psyche-centralized-server"])
                .status()
                .ok()
                .and_then(|s| s.success().then_some(()))
                .expect("Failed to build server");

            Command::new("cargo")
                .args(["build", "--release", "-p", "psyche-centralized-client"])
                .status()
                .ok()
                .and_then(|s| s.success().then_some(()))
                .expect("Failed to build client");

            let validate_cmd = if data_path.exists() {
                vec![
                    "run",
                    "--release",
                    "-p",
                    "psyche-centralized-server",
                    "validate-config",
                    "--state",
                    state_path.to_str().unwrap(),
                    "--data-config",
                    data_path.to_str().unwrap(),
                ]
            } else {
                vec![
                    "run",
                    "--release",
                    "-p",
                    "psyche-centralized-server",
                    "validate-config",
                    "--state",
                    state_path.to_str().unwrap(),
                ]
            };
            // Validate config
            let mut validate_cmd_builder = Command::new("cargo");
            validate_cmd_builder.args(validate_cmd);

            // Propagate DYLD_LIBRARY_PATH for torch library loading on macOS
            if let Ok(dyld_path) = std::env::var("DYLD_LIBRARY_PATH") {
                validate_cmd_builder.env("DYLD_LIBRARY_PATH", dyld_path);
            }

            validate_cmd_builder
                .status()
                .ok()
                .and_then(|s| s.success().then_some(()))
                .expect("Failed to validate config");

            let run_id = extract_run_id(&state_path)?;

            if start_args.headless {
                if start_args.random_kill_num.is_some() {
                    bail!("--random-kill-num is not supported with --headless yet");
                }
                run_headless(&start_args, &state_path, &data_path, &run_id)?;
                return Ok(());
            }

            ensure_tmux_available()?;
            ensure_port_available(start_args.server_port)?;

            // Create tmux session
            Command::new("tmux")
                .args(["new-session", "-d", "-s", "psyche"])
                .status()
                .ok()
                .and_then(|s| s.success().then_some(()))
                .expect("Failed to create tmux session");

            // Split windows and set up panes
            Command::new("tmux")
                .args(["split-window", "-h"])
                .status()
                .ok()
                .and_then(|s| s.success().then_some(()))
                .expect("Failed to split window horizontally");

            Command::new("tmux")
                .args(["select-pane", "-t", "0"])
                .status()
                .ok()
                .and_then(|s| s.success().then_some(()))
                .expect("Failed to select pane");

            Command::new("tmux")
                .args(["split-window", "-v"])
                .status()
                .ok()
                .and_then(|s| s.success().then_some(()))
                .expect("Failed to split window vertically");

            // Split remaining panes for clients
            Command::new("tmux")
                .args(["select-pane", "-t", "2"])
                .status()
                .ok()
                .and_then(|s| s.success().then_some(()))
                .expect("Failed to select pane");

            for _ in 1..start_args.num_clients {
                Command::new("tmux")
                    .args(["split-window", "-v"])
                    .status()
                    .ok()
                    .and_then(|s| s.success().then_some(()))
                    .expect("Failed to split window for client");
            }

            let start_time = OffsetDateTime::now_utc();

            // Start server
            let mut server_cmd = format!(
                "RUST_LOG={} cargo run -p psyche-centralized-server run --state {} --server-port {} --tui {}",
                start_args.log,
                state_path.display(),
                start_args.server_port,
                start_args.tui
            );
            if data_path.exists() {
                server_cmd.push_str(&format!(" --data-config {}", data_path.display()));
            }
            if start_args.write_log {
                let log_dir = format!(
                    "./logs/{}",
                    start_time
                        .format(format_description!(
                            "[year]-[month]-[day]_[hour]:[minute]:[second]"
                        ))
                        .unwrap()
                );
                std::fs::create_dir_all(&log_dir).unwrap();
                server_cmd.push_str(&format!(" --write-log {log_dir}/server.txt"));
            }

            println!("starting server: {server_cmd:?}");

            Command::new("tmux")
                .args(["select-pane", "-t", "0"])
                .status()
                .ok()
                .and_then(|s| s.success().then_some(()))
                .expect("Failed to select server pane");

            Command::new("tmux")
                .args(["send-keys", &server_cmd, "C-m"])
                .status()
                .ok()
                .and_then(|s| s.success().then_some(()))
                .expect("Failed to send server command");

            println!("Waiting for server startup...");
            wait_for_port_bind(start_args.server_port)?;
            println!("Server started!");

            // Start nvtop
            if !start_args.no_nvtop {
                if Command::new("tmux")
                    .args(["select-pane", "-t", "1"])
                    .status()
                    .ok()
                    .and_then(|s| s.success().then_some(()))
                    .is_some()
                {
                    let started = Command::new("tmux")
                        .args(["send-keys", "nvtop", "C-m"])
                        .status()
                        .ok()
                        .and_then(|s| s.success().then_some(()))
                        .is_some();
                    if !started {
                        eprintln!("Warning: failed to start `nvtop`; continuing");
                    }
                } else {
                    eprintln!("Warning: failed to select monitor pane; continuing");
                }
            }

            // Start clients
            for i in 2..=start_args.num_clients + 1 {
                start_client(&start_args, i, &run_id, true, start_time);
            }

            // // Attach to tmux session
            let mut tmux_session = Command::new("tmux")
                .args(["attach-session", "-t", "psyche"])
                .spawn()?;

            if let Some(kill_num) = start_args.random_kill_num {
                let allowed_to_kill = |item: &usize| {
                    if start_args.allowed_to_kill.is_empty() {
                        true
                    } else {
                        start_args.allowed_to_kill.contains(&(item - 1))
                    }
                };
                let mut last_kill_time = Instant::now();
                let kill_interval = Duration::from_secs(start_args.random_kill_interval);
                loop {
                    std::thread::sleep(Duration::from_millis(500));
                    if Instant::now() > (last_kill_time + kill_interval) {
                        last_kill_time = Instant::now();

                        let to_kill = {
                            let mut client_nums: Vec<usize> = (2..=start_args.num_clients + 1)
                                .filter(allowed_to_kill)
                                .collect();

                            client_nums.shuffle(&mut rand::rng());

                            client_nums.truncate(kill_num);
                            client_nums
                        };
                        for kill in to_kill {
                            Command::new("tmux")
                                .args(["select-pane", "-t", &kill.to_string()])
                                .status()
                                .ok()
                                .and_then(|s| s.success().then_some(()))
                                .expect("Failed to select client pane");
                            // send ctrl-c
                            Command::new("tmux")
                                .args(["send-keys", "-t", &kill.to_string(), "C-c"])
                                .status()
                                .ok()
                                .and_then(|s| s.success().then_some(()))
                                .expect("Failed to kill client");
                            // restart client
                            start_client(&start_args, kill, &run_id, false, start_time);
                        }
                    }

                    if tmux_session.try_wait().unwrap().is_some() {
                        break;
                    }
                }
            }

            let _ = tmux_session.wait(); // to prevent weird async tmux overlap with normal shell

            // failsafe kill
            Command::new("tmux")
                .args(["kill-session", "-t", "psyche"])
                .status()
                .expect("Failed to kill tmux session");

            Ok(())
        }
        Commands::PrintAllHelp { markdown } => {
            // This is a required argument for the time being.
            assert!(markdown);

            let () = clap_markdown::print_help_markdown::<Args>();

            Ok(())
        }
    }
}

fn ensure_tmux_available() -> Result<()> {
    match Command::new("tmux")
        .arg("-V")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
    {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => bail!(
            "`tmux` is required for non-headless mode (exit {}). Install `tmux` or pass `--headless`.",
            status
        ),
        Err(err) => bail!(
            "`tmux` is required for non-headless mode, but it could not be executed: {err}. Install `tmux` or pass `--headless`."
        ),
    }
}

fn run_headless(args: &StartArgs, state_path: &PathBuf, data_path: &PathBuf, run_id: &str) -> Result<()> {
    if let Some(exit_after) = args.headless_exit_after_secs {
        if exit_after == 0 {
            bail!("--headless-exit-after-secs must be > 0");
        }
    }
    ensure_port_available(args.server_port)?;

    let workspace_root = workspace_root()?;
    let target_dir = std::env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| workspace_root.join("target"));

    let server_bin = target_dir.join("release").join("psyche-centralized-server");
    let client_bin = target_dir.join("release").join("psyche-centralized-client");
    if !server_bin.exists() {
        bail!(
            "Server binary not found at {} (did `cargo build --release -p psyche-centralized-server` succeed?)",
            server_bin.display()
        );
    }
    if !client_bin.exists() {
        bail!(
            "Client binary not found at {} (did `cargo build --release -p psyche-centralized-client` succeed?)",
            client_bin.display()
        );
    }

    let start_time = OffsetDateTime::now_utc();
    let mut processes = ChildProcesses { children: Vec::new() };

    // Start server
    let mut server_cmd = Command::new(&server_bin);
    server_cmd
        .env("RUST_LOG", &args.log);

    // Propagate DYLD_LIBRARY_PATH for torch library loading on macOS
    if let Ok(dyld_path) = std::env::var("DYLD_LIBRARY_PATH") {
        server_cmd.env("DYLD_LIBRARY_PATH", dyld_path);
    }

    server_cmd.args([
            "run",
            "--state",
            state_path.to_str().unwrap(),
            "--server-port",
            &args.server_port.to_string(),
            "--tui",
            &args.tui.to_string(),
        ])
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    if data_path.exists() {
        server_cmd.args(["--data-config", data_path.to_str().unwrap()]);
    }
    if args.write_log {
        let log_dir = format!(
            "./logs/{}",
            start_time
                .format(format_description!(
                    "[year]-[month]-[day]_[hour]:[minute]:[second]"
                ))
                .unwrap()
        );
        std::fs::create_dir_all(&log_dir)?;
        server_cmd.args(["--write-log", &format!("{log_dir}/server.txt")]);
    }

    println!("starting server: {server_cmd:?}");
    let server_child = server_cmd.spawn().context("Failed to start server")?;
    processes.children.push(server_child);

    println!("Waiting for server startup...");
    wait_for_server_startup(args.server_port, processes.children.first_mut().unwrap())?;
    println!("Server started!");

    // Start clients
    for i in 2..=args.num_clients + 1 {
        let client_child = spawn_client_headless(args, &client_bin, i, run_id, start_time)?;
        processes.children.push(client_child);
    }

    // Wait until the server exits.
    if let Some(exit_after) = args.headless_exit_after_secs {
        let deadline = Instant::now() + Duration::from_secs(exit_after);
        loop {
            if let Some(status) = processes.children.first_mut().unwrap().try_wait()? {
                println!("server exited with {status}");
                break;
            }
            if Instant::now() >= deadline {
                println!("headless duration elapsed, shutting down testnet...");
                break;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    } else {
        let status = processes.children.first_mut().unwrap().wait()?;
        println!("server exited with {status}");
    }

    Ok(())
}

fn wait_for_server_startup(port: u16, server_child: &mut Child) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if port_is_bound(port)? {
            return Ok(());
        }
        if let Some(status) = server_child.try_wait()? {
            bail!("Server exited before binding port {port}: {status}");
        }
        if Instant::now() > deadline {
            bail!("Timed out waiting for server to bind port {port}");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn wait_for_port_bind(port: u16) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if port_is_bound(port)? {
            return Ok(());
        }
        if Instant::now() > deadline {
            bail!("Timed out waiting for server to bind port {port}");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn port_is_bound(port: u16) -> Result<bool> {
    let addr = format!("0.0.0.0:{port}");
    match std::net::TcpListener::bind(&addr) {
        Ok(listener) => {
            drop(listener);
            Ok(false)
        }
        Err(err) if err.kind() == std::io::ErrorKind::AddrInUse => Ok(true),
        Err(err) => bail!("Failed checking port {port}: {err}"),
    }
}

fn spawn_client_headless(
    args: &StartArgs,
    client_bin: &PathBuf,
    i: usize,
    run_id: &str,
    start_time: OffsetDateTime,
) -> Result<Child> {
    let raw_key = format!("{:0>64x}", i - 1);
    let metrics_local_port = 6269 + i - 1;
    let client_idx = i.saturating_sub(2);
    let matformer_tier = client_matformer_tier(args, client_idx);
    let helper_fraction = client_matformer_helper_fraction(args, client_idx);
    let rotation_interval = client_matformer_helper_rotation_interval(args, client_idx);

    let logs_mode = if args.tui { "tui" } else { "console" };
    let mut cmd = Command::new(client_bin);
    cmd.env("METRICS_LOCAL_PORT", metrics_local_port.to_string())
        .env("RUST_LOG", &args.log)
        .env("RUST_BACKTRACE", "1")
        .env("RAW_IDENTITY_SECRET_KEY", raw_key);

    // Propagate DYLD_LIBRARY_PATH for torch library loading on macOS
    if let Ok(dyld_path) = std::env::var("DYLD_LIBRARY_PATH") {
        cmd.env("DYLD_LIBRARY_PATH", dyld_path);
    }

    cmd
        .args([
            "train",
            "--run-id",
            run_id,
            "--server-addr",
            &format!("localhost:{}", args.server_port),
            "--logs",
            logs_mode,
            "--device",
            &args.client_device,
            "--matformer-tier",
            &matformer_tier.to_string(),
            "--matformer-helper-fraction",
            &helper_fraction.to_string(),
            "--matformer-helper-rotation-interval",
            &rotation_interval.to_string(),
            "--iroh-discovery",
            &args.client_iroh_discovery,
            "--iroh-relay",
            &args.client_iroh_relay,
        ])
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    if args.matformer_distillation_beta_max > 0.0 {
        cmd.args([
            "--matformer-distillation-beta-max",
            &args.matformer_distillation_beta_max.to_string(),
            "--matformer-distillation-warmup-steps",
            &args.matformer_distillation_warmup_steps.to_string(),
            "--matformer-distillation-start-step",
            &args.matformer_distillation_start_step.to_string(),
            "--matformer-distillation-top-k",
            &args.matformer_distillation_top_k.to_string(),
            "--matformer-distillation-temperature",
            &args.matformer_distillation_temperature.to_string(),
        ]);
    }

    if let Some(token) = &args.hf_token {
        cmd.env("HF_TOKEN", token);
    }

    if let Some(dir) = &args.write_distro_data {
        cmd.args(["--write-gradients-dir", dir.to_str().unwrap()]);
    }

    if let Some(repo) = &args.first_client_checkpoint {
        if i == 2 {
            cmd.args(["--checkpoint-dir", "./checkpoints", "--hub-repo", repo]);
        }
    }

    if let Some(entity) = &args.wandb_entity {
        cmd.args(["--wandb-entity", entity]);
    }
    if let Some(group) = &args.wandb_group {
        cmd.args(["--wandb-group", group]);
    }
    if let Some(project) = &args.wandb_project {
        cmd.args(["--wandb-project", project]);
    }
    if args.wandb_step_logging {
        cmd.arg("--wandb-step-logging");
    }
    if args.wandb_system_metrics {
        cmd.arg("--wandb-system-metrics");
        cmd.args([
            "--wandb-system-metrics-interval-secs",
            &args.wandb_system_metrics_interval_secs.to_string(),
        ]);
    }

    if args.write_log {
        let log_dir = format!(
            "./logs/{}",
            start_time
                .format(format_description!(
                    "[year]-[month]-[day]_[hour]:[minute]:[second]"
                ))
                .unwrap()
        );
        std::fs::create_dir_all(&log_dir)?;
        cmd.args(["--write-log", &format!("{log_dir}/client-{}.txt", i - 1)]);
    }

    if let Some(s) = args.optim_stats {
        cmd.args(["--optim-stats", &s.to_string()]);
    }

    if let Some(evals) = &args.eval_tasks {
        cmd.args(["--eval-tasks", evals]);
    }

    if let Some(dir) = &args.client_checkpoint_dir {
        cmd.args(["--checkpoint-dir", dir.to_str().unwrap()]);
        cmd.args(["--keep-steps", &args.client_keep_steps.to_string()]);
    }

    // Fault injection flags
    if let Some(latency) = &args.fault_latency_ms {
        cmd.args(["--fault-latency-ms", latency]);
    }
    if let Some(loss) = args.fault_packet_loss {
        cmd.args(["--fault-packet-loss", &loss.to_string()]);
    }
    if let Some(limit) = args.fault_bandwidth_limit {
        cmd.args(["--fault-bandwidth-limit", &limit.to_string()]);
    }
    if let Some(seed) = args.fault_seed {
        // Use different seed per client for varied behavior, but deterministic
        cmd.args(["--fault-seed", &(seed + i as u64).to_string()]);
    }

    println!("starting client {i}: {cmd:?}");
    let child = cmd.spawn().with_context(|| format!("Failed to start client {i}"))?;
    Ok(child)
}

fn workspace_root() -> Result<PathBuf> {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if dir.join("Cargo.lock").exists() {
            return Ok(dir);
        }
        if !dir.pop() {
            bail!("Failed to locate workspace root (Cargo.lock not found)");
        }
    }
}

fn ensure_port_available(port: u16) -> Result<()> {
    let addr = format!("0.0.0.0:{port}");
    match std::net::TcpListener::bind(&addr) {
        Ok(listener) => {
            drop(listener);
            Ok(())
        }
        Err(err) => bail!("Server port {port} is not available: {err}"),
    }
}

fn start_client(
    args: &StartArgs,
    i: usize,
    run_id: &String,
    print: bool,
    start_time: OffsetDateTime,
) {
    // hex 1, 2, 3, etc.
    let raw_key = format!("{:0>64x}", i - 1);
    let client_idx = i.saturating_sub(2);
    let matformer_tier = client_matformer_tier(args, client_idx);
    let helper_fraction = client_matformer_helper_fraction(args, client_idx);
    let rotation_interval = client_matformer_helper_rotation_interval(args, client_idx);

    Command::new("tmux")
        .args(["select-pane", "-t", &i.to_string()])
        .status()
        .ok()
        .and_then(|s| s.success().then_some(()))
        .expect("Failed to select client pane");

    let mut cmd: OsString = if let Some(token) = &args.hf_token {
        format!("HF_TOKEN={token} ").into()
    } else {
        OsString::new()
    };

    let metrics_local_port = 6269 + i - 1;

    cmd.push(format!(
        "METRICS_LOCAL_PORT={metrics_local_port} RUST_LOG={} RUST_BACKTRACE=1 RAW_IDENTITY_SECRET_KEY={} cargo run -p psyche-centralized-client train --run-id {} --server-addr localhost:{} --logs {} --device {} --matformer-tier {} --matformer-helper-fraction {} --matformer-helper-rotation-interval {} --iroh-discovery {} --iroh-relay {}",
        args.log,
        raw_key,
        run_id,
        args.server_port,
        if args.tui {
            "tui"
        } else {
            "console"
        },
        args.client_device,
        matformer_tier,
        helper_fraction,
        rotation_interval,
        args.client_iroh_discovery,
        args.client_iroh_relay,
    ));

    if args.matformer_distillation_beta_max > 0.0 {
        cmd.push(format!(
            " --matformer-distillation-beta-max {} --matformer-distillation-warmup-steps {} --matformer-distillation-start-step {} --matformer-distillation-top-k {} --matformer-distillation-temperature {}",
            args.matformer_distillation_beta_max,
            args.matformer_distillation_warmup_steps,
            args.matformer_distillation_start_step,
            args.matformer_distillation_top_k,
            args.matformer_distillation_temperature,
        ));
    }

    if let Some(dir) = &args.write_distro_data {
        cmd.push(" --write-gradients-dir ");
        cmd.push(dir);
    }

    if let Some(repo) = &args.first_client_checkpoint {
        if i == 2 {
            cmd.push(format!(" --checkpoint-dir ./checkpoints --hub-repo {repo}"));
        }
    }

    if let Some(entity) = &args.wandb_entity {
        cmd.push(format!(" --wandb-entity {entity}"));
    }
    if let Some(group) = &args.wandb_group {
        cmd.push(format!(" --wandb-group {group}"));
    }
    if let Some(project) = &args.wandb_project {
        cmd.push(format!(" --wandb-project {project}"));
    }
    if args.wandb_step_logging {
        cmd.push(" --wandb-step-logging".to_string());
    }
    if args.wandb_system_metrics {
        cmd.push(format!(
            " --wandb-system-metrics --wandb-system-metrics-interval-secs {}",
            args.wandb_system_metrics_interval_secs
        ));
    }

    if args.write_log {
        let log_dir = format!(
            "./logs/{}",
            start_time
                .format(format_description!(
                    "[year]-[month]-[day]_[hour]:[minute]:[second]"
                ))
                .unwrap()
        );
        std::fs::create_dir_all(&log_dir).unwrap();
        cmd.push(format!(" --write-log {log_dir}/client-{}.txt", i - 1))
    }

    if let Some(s) = args.optim_stats {
        cmd.push(format!(" --optim-stats {s}"));
    }

    if let Some(evals) = &args.eval_tasks {
        cmd.push(format!(" --eval-tasks {evals}"))
    }

    if let Some(dir) = &args.client_checkpoint_dir {
        cmd.push(format!(" --checkpoint-dir {} --keep-steps {}", dir.display(), args.client_keep_steps));
    }

    // Fault injection flags
    if let Some(latency) = &args.fault_latency_ms {
        cmd.push(format!(" --fault-latency-ms {latency}"));
    }
    if let Some(loss) = args.fault_packet_loss {
        cmd.push(format!(" --fault-packet-loss {loss}"));
    }
    if let Some(limit) = args.fault_bandwidth_limit {
        cmd.push(format!(" --fault-bandwidth-limit {limit}"));
    }
    if let Some(seed) = args.fault_seed {
        // Use different seed per client for varied behavior, but deterministic
        cmd.push(format!(" --fault-seed {}", seed + i as u64));
    }

    if print {
        println!("starting client {i}: {cmd:?}");
    }

    Command::new("tmux")
        .args([OsString::from("send-keys"), cmd, OsString::from("C-m")])
        .status()
        .ok()
        .and_then(|s| s.success().then_some(()))
        .expect("Failed to send server command");
}

fn client_matformer_tier(args: &StartArgs, client_zero_based_index: usize) -> u8 {
    match args.client_matformer_tiers.as_slice() {
        [] => 0,
        [tier] => *tier,
        tiers => tiers[client_zero_based_index % tiers.len()],
    }
}

fn client_matformer_helper_fraction(args: &StartArgs, client_zero_based_index: usize) -> f32 {
    match args.client_matformer_helper_fractions.as_slice() {
        [] => 0.0,
        [fraction] => *fraction,
        fractions => fractions[client_zero_based_index % fractions.len()],
    }
}

fn client_matformer_helper_rotation_interval(
    args: &StartArgs,
    client_zero_based_index: usize,
) -> u64 {
    match args.client_matformer_helper_rotation_intervals.as_slice() {
        [] => 16,
        [interval] => *interval,
        intervals => intervals[client_zero_based_index % intervals.len()],
    }
}
