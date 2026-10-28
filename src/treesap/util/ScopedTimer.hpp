#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>

namespace treesap {

class ScopedTimer {
    using Duration = std::chrono::steady_clock::duration;
    using TimePoint = std::chrono::steady_clock::time_point;

  public:
    explicit ScopedTimer(std::string name) : timer_name(std::move(name)), start(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        Duration elapsed = std::chrono::steady_clock::now() - start;

        // update aggregated stats
        auto& [total_duration, count] = totals_counts[timer_name];
        total_duration += elapsed;
        count++;

        Duration avg_duration = total_duration / count;

        // print formatted summary
        std::cout << "[Timer: " << timer_name << "] "
                  << "Run: " << formatDuration(elapsed) << " | "
                  << "Total: " << formatDuration(total_duration) << " | "
                  << "Avg: " << formatDuration(avg_duration) << " (" << count << " runs)\n";
    }

  private:
    std::string timer_name;
    TimePoint start;

    // convert any chrono duration to a readable string (µs, ms, or s)
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
    inline static std::unordered_map<std::string, std::pair<Duration, size_t>> totals_counts;
};

} // namespace treesap