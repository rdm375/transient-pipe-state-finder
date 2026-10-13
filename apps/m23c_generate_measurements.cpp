#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ReferenceSample {
    double time = 0.0;
    double inlet_pressure = 0.0;
    double inlet_flow = 0.0;
    double outlet_pressure = 0.0;
    double outlet_flow = 0.0;
};

std::vector<ReferenceSample> read_reference(
    const std::filesystem::path& path)
{
    std::ifstream input(path);

    if (!input)
        throw std::runtime_error(
            "Cannot open reference trajectory: " +
            path.string());

    std::string header;

    if (!std::getline(input, header))
        throw std::runtime_error("Empty reference CSV");

    const std::string expected_header =
        "time_s,inlet_pressure_Pa,"
        "inlet_flow_kg_s,"
        "outlet_pressure_Pa,"
        "outlet_flow_kg_s";

    if (header != expected_header)
        throw std::runtime_error(
            "Unexpected reference CSV schema");

    std::vector<ReferenceSample> samples;

    std::string line;

    while (std::getline(input, line)) {
        if (line.empty())
            continue;

        std::stringstream stream(line);

        std::string field;
        double values[5];

        for (int j = 0; j < 5; ++j) {
            if (!std::getline(stream, field, ','))
                throw std::runtime_error(
                    "Malformed reference CSV row");

            std::size_t consumed = 0;

            values[j] = std::stod(field, &consumed);

            if (consumed != field.size() ||
                !std::isfinite(values[j]))
                throw std::runtime_error(
                    "Invalid reference CSV value");
        }

        if (std::getline(stream, field, ','))
            throw std::runtime_error(
                "Extra reference CSV column");

        samples.push_back({
            values[0],
            values[1],
            values[2],
            values[3],
            values[4]
        });
    }

    if (samples.empty())
        throw std::runtime_error(
            "Reference trajectory has no samples");

    return samples;
}

void validate_reference(
    const std::vector<ReferenceSample>& samples,
    double expected_dt)
{
    if (samples.size() != 61)
        throw std::runtime_error(
            "Expected 61 reference samples");

    for (std::size_t k = 0; k < samples.size(); ++k) {
        const auto& sample = samples[k];

        const double expected_time =
            static_cast<double>(k)*expected_dt;

        if (std::abs(sample.time-expected_time) > 1.0e-9)
            throw std::runtime_error(
                "Unexpected reference sampling time");

        if (sample.inlet_pressure <= 0.0 ||
            sample.outlet_pressure <= 0.0)
            throw std::runtime_error(
                "Nonphysical reference pressure");
    }
}

} // namespace

int main()
{
    try {
        constexpr double dt = 10.0;

        constexpr double sigma_flow = 0.1;
        constexpr double sigma_pressure = 1000.0;

        constexpr std::uint64_t seed = 20261013;

        const auto reference = read_reference(
            "benchmarks/results-m23a/reference_boundary.csv");

        validate_reference(reference, dt);

        std::filesystem::create_directories(
            "benchmarks/results-m23c");

        std::mt19937_64 rng(seed);

        std::normal_distribution<double> standard_normal(
            0.0, 1.0);

        const auto csv_path =
            "benchmarks/results-m23c/measurements.csv";

        std::ofstream csv(csv_path);

        if (!csv)
            throw std::runtime_error(
                "Cannot create measurements CSV");

        csv << std::setprecision(17);

        csv
            << "time_s,"
            << "inlet_pressure_control_Pa,"
            << "outlet_flow_control_kg_s,"
            << "inlet_flow_truth_kg_s,"
            << "inlet_flow_observed_kg_s,"
            << "outlet_pressure_truth_Pa,"
            << "outlet_pressure_observed_Pa,"
            << "inlet_flow_noise_kg_s,"
            << "outlet_pressure_noise_Pa\n";

        double sum_flow_noise = 0.0;
        double sum_pressure_noise = 0.0;

        double sum_flow_noise_squared = 0.0;
        double sum_pressure_noise_squared = 0.0;

        for (const auto& sample : reference) {
            const double flow_noise =
                sigma_flow*standard_normal(rng);

            const double pressure_noise =
                sigma_pressure*standard_normal(rng);

            const double observed_flow =
                sample.inlet_flow + flow_noise;

            const double observed_pressure =
                sample.outlet_pressure + pressure_noise;

            sum_flow_noise += flow_noise;
            sum_pressure_noise += pressure_noise;

            sum_flow_noise_squared +=
                flow_noise*flow_noise;

            sum_pressure_noise_squared +=
                pressure_noise*pressure_noise;

            csv
                << sample.time << ','
                << sample.inlet_pressure << ','
                << sample.outlet_flow << ','
                << sample.inlet_flow << ','
                << observed_flow << ','
                << sample.outlet_pressure << ','
                << observed_pressure << ','
                << flow_noise << ','
                << pressure_noise << '\n';
        }

        csv.close();

        if (!csv)
            throw std::runtime_error(
                "Failed writing measurements CSV");

        const auto metadata_path =
            "benchmarks/results-m23c/measurement_metadata.json";

        std::ofstream metadata(metadata_path);

        if (!metadata)
            throw std::runtime_error(
                "Cannot create measurement metadata");

        metadata << std::setprecision(17);

        metadata
            << "{\n"
            << "  \"experiment\": \"M2.3-C\",\n"
            << "  \"description\": "
               "\"Synthetic boundary measurements\",\n"
            << "  \"reference\": "
               "\"benchmarks/results-m23a/reference_boundary.csv\",\n"
            << "  \"sample_count\": " << reference.size() << ",\n"
            << "  \"sampling_interval_s\": " << dt << ",\n"
            << "  \"duration_s\": "
            << reference.back().time << ",\n"
            << "  \"random_seed\": " << seed << ",\n"
            << "  \"random_engine\": \"std::mt19937_64\",\n"
            << "  \"normal_distribution\": "
               "\"std::normal_distribution<double>\",\n"
            << "  \"noise_model\": \"independent Gaussian\",\n"
            << "  \"inlet_flow_sigma_kg_s\": "
            << sigma_flow << ",\n"
            << "  \"outlet_pressure_sigma_Pa\": "
            << sigma_pressure << ",\n"
            << "  \"known_controls\": [\n"
            << "    \"inlet_pressure\",\n"
            << "    \"outlet_mass_flow\"\n"
            << "  ],\n"
            << "  \"observed_channels\": [\n"
            << "    \"inlet_mass_flow\",\n"
            << "    \"outlet_pressure\"\n"
            << "  ]\n"
            << "}\n";

        metadata.close();

        if (!metadata)
            throw std::runtime_error(
                "Failed writing measurement metadata");

        const double count =
            static_cast<double>(reference.size());

        const double flow_mean =
            sum_flow_noise/count;

        const double pressure_mean =
            sum_pressure_noise/count;

        const double flow_rms =
            std::sqrt(sum_flow_noise_squared/count);

        const double pressure_rms =
            std::sqrt(sum_pressure_noise_squared/count);

        std::cout << std::setprecision(12)
            << "experiment=M2.3-C\n"
            << "samples=" << reference.size() << '\n'
            << "sampling_interval_s=" << dt << '\n'
            << "random_seed=" << seed << '\n'
            << "sigma_flow_kg_s=" << sigma_flow << '\n'
            << "sigma_pressure_Pa=" << sigma_pressure << '\n'
            << "flow_noise_mean_kg_s="
            << flow_mean << '\n'
            << "pressure_noise_mean_Pa="
            << pressure_mean << '\n'
            << "flow_noise_rms_kg_s="
            << flow_rms << '\n'
            << "pressure_noise_rms_Pa="
            << pressure_rms << '\n'
            << "PASS M2.3-C synthetic measurements\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "FAIL M2.3-C: "
            << error.what() << '\n';

        return 1;
    }
}
