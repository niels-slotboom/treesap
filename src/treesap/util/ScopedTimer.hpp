// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

/**
 * @file
 * @brief Provides an RAII-based scope timer and performance statistics aggregator for AMReX.
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include <AMReX_Print.H>

namespace treesap {

/**
 * @brief RAII-based timer which measures execution time of scopes and tracks aggregated statistics.
 *
 * `ScopedTimer` is meant to be instantiated with a `std::string` timer name, causing it to store the current time as
 * the scope start time. Its destruction marks the stop time, which triggers the computation of the scope duration, the
 * printing of diagnostics, and the logging of aggregate statistics under the same timer name. At any time, the
 * aggregate statistics can be printed to `amrex::AllPrint()` as a sorted overview.
 *
 * The static member `ScopedTimer::enabled` dictates whether the above behaviour is enabled or disabled.
 */
class ScopedTimer {
    using Duration = std::chrono::steady_clock::duration;
    using TimePoint = std::chrono::steady_clock::time_point;

    /**
     * @brief Holds accumulated timing statistics for a named timer.
     */
    struct TimerStats {
        Duration total_duration{0};  ///< Cumulative duration across all runs.
        double sum_squares_us = 0.0; ///< Sum of squared durations in microseconds for variance computation.
        size_t count = 0;            ///< Total number of executions.
    };

  public:
    /**
     * @brief Initializes a timer with a name and records the start time.
     * @param name Unique identifier for the timer.
     */
    explicit ScopedTimer(std::string name) : timer_name(std::move(name)), start(std::chrono::steady_clock::now()) {}

    /**
     * @brief Measures elapsed time on scope exit, updates aggregate statistics, and prints run details.
     */
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
        amrex::AllPrint() << "[Timer: " << timer_name << "] "
                          << "Run: " << formatDuration(elapsed) << " | "
                          << "Total: " << formatDuration(stats.total_duration) << " | "
                          << "Avg: " << formatDuration(avg_duration) << " | "
                          << "StdDev: " << formatDuration(stdev_duration) << " (" << stats.count << " runs)\n";
    }

    /**
     * @brief Prints a sorted profiling overview table with left-aligned timer names and clean spacing.
     *
     * The overview lists the total execution count, total execution time, average execution time and standard deviation
     * of each named timer that was used one or multiple times up to the point of calling `printOverview()`.
     */
    static void printOverview() {
        amrex::AllPrint() << "--- --- --- ScopedTimer Profiling Overview --- --- ---" << std::endl;

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
    std::string timer_name; ///< Name assigned to the current timer instance.
    TimePoint start;        ///< Start time point of the current timer instance.

    /**
     * @brief Converts any duration to a readable text string using micro, milli, or second units with one decimal
     * place.
     * @tparam Rep Arithmetic type representing the number of ticks.
     * @tparam Period A std::ratio representing the tick period.
     * @param d Duration value to format.
     * @return Formatted string representation of the duration.
     */
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

    inline static std::unordered_map<std::string, TimerStats>
        totals_counts; ///< Global map storing aggregated statistics for all timers.

  public:
    inline static bool enabled = false; ///< Global switch to enable or disable timer logging and collection.
};

} // namespace treesap