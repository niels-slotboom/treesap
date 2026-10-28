#pragma once

#include <chrono>
#include <cmath>
#include <iostream>
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

  private:
    std::string timer_name;
    TimePoint start;

    /// @brief Convert any duration to a readable text string using micro, milli, or second units.
    template <typename Rep, typename Period> static std::string formatDuration(std::chrono::duration<Rep, Period> d) {
        auto duration_ms = std::chrono::duration<double, std::milli>(d).count();

        if (duration_ms < 1.0) {
            auto duration_us = std::chrono::duration<double, std::micro>(d).count();
            return std::to_string(duration_us) + " µs";
        } else if (duration_ms >= 1000.0) {
            auto duration_s = std::chrono::duration<double>(d).count();
            return std::to_string(duration_s) + " s";
        } else {
            return std::to_string(duration_ms) + " ms";
        }
    }

    inline static std::unordered_map<std::string, TimerStats> totals_counts;

  public:
    inline static bool enabled = false;
};

} // namespace treesap