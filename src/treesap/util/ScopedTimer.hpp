#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace treesap {

class ScopedTimer {
    using Duration = std::chrono::steady_clock::duration;
    using TimePoint = std::chrono::steady_clock::time_point;

    /// @brief Holds accumulated timing statistics for a named timer.
    struct TimerStats {
        Duration total_duration{0};
        double sum_squares_us = 0.0;
        size_t count = 0;
    };

  public:
    explicit ScopedTimer(std::string name) : timer_name(std::move(name)), start(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        if (!enabled)
            return;

        Duration elapsed = std::chrono::steady_clock::now() - start;
        double elapsed_us = std::chrono::duration<double, std::micro>(elapsed).count();

        // update aggregated stats
        auto& stats = totals_counts[timer_name];
        stats.total_duration += elapsed;
        stats.sum_squares_us += elapsed_us * elapsed_us;
        stats.count++;

        Duration avg_duration = stats.total_duration / stats.count;

        // compute standard deviation in microseconds
        double mean_us = std::chrono::duration<double, std::micro>(stats.total_duration).count() / stats.count;
        double variance_us = (stats.sum_squares_us / stats.count) - (mean_us * mean_us);
        double stdev_us = variance_us > 0.0 ? std::sqrt(variance_us) : 0.0;
        auto stdev_duration = std::chrono::duration<double, std::micro>(stdev_us);

        // print formatted summary including standard deviation
        std::cout << "[Timer: " << timer_name << "] "
                  << "Run: " << formatDuration(elapsed) << " | "
                  << "Total: " << formatDuration(stats.total_duration) << " | "
                  << "Avg: " << formatDuration(avg_duration) << " | "
                  << "StdDev: " << formatDuration(stdev_duration) << " (" << stats.count << " runs)\n";
    }

    /**
     * @brief Prints a sorted profiling overview table with left-aligned timer names and clean spacing.
     */
    static void printOverview() {
        std::cout << "--- --- --- ScopedTimer Profiling Overview --- --- ---" << std::endl;

        std::vector<std::pair<std::string, TimerStats>> sorted_entries(totals_counts.begin(), totals_counts.end());

        std::ranges::sort(sorted_entries, [](const auto& a, const auto& b) {
            return a.second.total_duration > b.second.total_duration;
        });

        for (const auto& [name, stats] : sorted_entries) {
            Duration avg_duration = stats.total_duration / stats.count;

            double mean_us = std::chrono::duration<double, std::micro>(stats.total_duration).count() / stats.count;
            double variance_us = (stats.sum_squares_us / stats.count) - (mean_us * mean_us);
            double stdev_us = variance_us > 0.0 ? std::sqrt(variance_us) : 0.0;
            auto stdev_duration = std::chrono::duration<double, std::micro>(stdev_us);

            std::cout << "[Timer: " << std::left << std::setw(35) << name << "] " << std::right
                      << "Executions: " << std::setw(6) << stats.count << " | "
                      << "Total: " << std::setw(8) << formatDuration(stats.total_duration) << " | "
                      << "Avg: " << std::setw(8) << formatDuration(avg_duration) << " | "
                      << "StdDev: " << std::setw(9) << formatDuration(stdev_duration) << std::endl;
        }
    }

  private:
    std::string timer_name;
    TimePoint start;

    /// @brief Convert any duration to a readable text string using micro, milli, or second units with one decimal
    /// place.
    template <typename Rep, typename Period> static std::string formatDuration(std::chrono::duration<Rep, Period> d) {
        auto duration_ms = std::chrono::duration<double, std::milli>(d).count();
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1);

        if (duration_ms < 1.0) {
            oss << std::chrono::duration<double, std::micro>(d).count() << " \u00b5s";
        } else if (duration_ms >= 1000.0) {
            oss << std::chrono::duration<double>(d).count() << " s";
        } else {
            oss << duration_ms << " ms";
        }
        return oss.str();
    }

    inline static std::unordered_map<std::string, TimerStats> totals_counts;

  public:
    inline static bool enabled = false;
};

} // namespace treesap