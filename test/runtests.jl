using MIMOPotts
using DataFrames
using Test

@testset "PAM levels" begin
    @test MIMOPotts.pam_levels("QPSK") == [-inv(sqrt(2.0)), inv(sqrt(2.0))]
    @test length(MIMOPotts.pam_levels("16QAM")) == 4
end

@testset "JLD2 result save/load" begin
    result = (
        runs = DataFrame(runId = [1], source = ["synthetic"], noiseRatio = [0.5]),
        trials = DataFrame(runId = [1], outerTrial = [1], bestDistance = [1.25]),
    )
    path = tempname() * ".jld2"
    saved = save_mimo_potts_results(path, result; metadata = (; note = "unit"))
    loaded = load_mimo_potts_results(saved; includeMetadata = true)
    @test loaded.format == :compact
    @test loaded.schemaVersion == 1
    @test loaded.metadata["note"] == "unit"
    @test loaded.result.runs == result.runs
    @test loaded.result.trials == result.trials
end
