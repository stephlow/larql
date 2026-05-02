use clap::{Args, Subcommand};

use super::capture::{run_capture, CaptureArgs};
use super::oracle::{
    run_oracle_lowrank, run_oracle_roundtrip, OracleLowrankArgs, OracleRoundtripArgs,
};
use super::oracle_pq::{run_oracle_pq, OraclePqArgs};
use super::sanity::{run_sanity_check, SanityCheckArgs};
use super::static_replace::{run_static_replace, StaticReplaceArgs};
use super::zero_ablate::{run_zero_ablate, ZeroAblateArgs};

#[derive(Args)]
pub struct OvRdArgs {
    #[command(subcommand)]
    command: OvRdCommand,
}

#[derive(Subcommand)]
enum OvRdCommand {
    /// Capture pre-W_O OV output statistics from a Q4K vindex.
    Capture(CaptureArgs),

    /// Gate 1: zero selected pre-W_O heads and measure final-logit KL.
    ZeroAblate(ZeroAblateArgs),

    /// Static replacement gate: zero/global/position/stratum pre-W_O means.
    StaticReplace(StaticReplaceArgs),

    /// Sanity checks for pre-W_O replacement and W_O block equivalence.
    SanityCheck(SanityCheckArgs),

    /// Oracle RD plumbing check: W_O-coordinate roundtrip with no truncation.
    OracleRoundtrip(OracleRoundtripArgs),

    /// Oracle RD: unquantized low-rank sweep in W_O-visible coordinates.
    OracleLowrank(OracleLowrankArgs),

    /// Oracle RD: oracle-addressed product quantization in PCA coordinates.
    OraclePq(OraclePqArgs),
}

pub fn run(args: OvRdArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.command {
        OvRdCommand::Capture(capture) => run_capture(capture),
        OvRdCommand::ZeroAblate(zero) => run_zero_ablate(zero),
        OvRdCommand::StaticReplace(static_replace) => run_static_replace(static_replace),
        OvRdCommand::SanityCheck(sanity) => run_sanity_check(sanity),
        OvRdCommand::OracleRoundtrip(roundtrip) => run_oracle_roundtrip(roundtrip),
        OvRdCommand::OracleLowrank(lowrank) => run_oracle_lowrank(lowrank),
        OvRdCommand::OraclePq(pq) => run_oracle_pq(pq),
    }
}
