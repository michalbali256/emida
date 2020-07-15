#pragma once

#include <chrono>
#include <array>
#include <unordered_map>

#include "common.hpp"

namespace emida
{

struct stats
{
	inline static size_t total_pics = 0;
	inline static size2_t border = { 0, 0 };
	vec2<std::vector<size_t>> histogram = {std::vector<size_t>(100), std::vector<size_t>(100) };

	template<typename T>
	void inc_histogram(const std::vector<vec2<T>>& offsets)
	{
		for (const auto& off : offsets)
		{
			if (abs(off.x) < histogram.x.size())
				++histogram.x[(size_t)abs(off.x)];
			if (abs(off.y) < histogram.y.size())
				++histogram.y[(size_t)abs(off.y)];
		}
	}

	void write_histogram()
	{
		for (size_t i = 0; i < histogram.x.size(); ++i)
			std::cerr << i << ": " << histogram.x[i] << "\n";
		std::cerr << "\n";
		for (size_t i = 0; i < histogram.y.size(); ++i)
			std::cerr << i << ": " << histogram.y[i] << "\n";
	}
};

class stopwatch
{
public:
	stopwatch()
		: stopwatch(true, 2){}

	stopwatch(bool activate, size_t levels = 2, int bonus_indent = 0)
		: active_(activate && global_activate)
		, start_(levels)
		, bonus_indent_(bonus_indent)
	{
		auto start = c.now();
		for (auto& s : start_)
			s = start;
	}
	
	void tick(const std::string& label)
	{
		tick(label, (int)start_.size() - 1);
	}

	void tick(const std::string& label, int level)
	{
		if (!active_)
			return;
		write_duration(label, start_[level], c.now(), level-1);
		auto start = c.now();
		for (size_t i = level; i < start_.size(); ++i)
			start_[i] = start;
	}

	void total()
	{
		if (!active_)
			return;
		write_duration(total_, start_[0], c.now(), 0);
		auto start = c.now();
		for (size_t i = 0; i < start_.size(); ++i)
			start_[i] = start;
	}

	void zero()
	{
		auto start = c.now();
		for (auto & s: start_)
			s = start;
	}
	
	inline static stats global_stats;
	inline static bool global_activate;

	static void write_durations()
	{
		for (auto& t: times_)
		{
			std::cerr << t.first << ": " << t.second.first / t.second.second << "ms\n";
		}
	}

private:
	inline static const std::string total_ = "TOTAL: ";
	
	std::vector<std::chrono::high_resolution_clock::time_point> start_;
	
	inline static std::unordered_map<std::string, std::pair<double, size_t>> times_;

	std::chrono::high_resolution_clock c;
	bool active_;
	int bonus_indent_;

	std::array<std::string, 6> indentation_ = {
		"",
		"  ",
		"    ",
		"      ",
		"        ",
		"          "
	};

	template<typename time_point>
	void write_duration(const std::string & label, time_point start, time_point end, int level)
	{
		auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		//std::cerr << indentation_[level < 0 ? 0 : level + bonus_indent_] << label << std::to_string(dur.count() / 1000.0) << " ms" << "\n";
		times_[label].first += dur.count() / 1000.0;
		++times_[label].second;

	}
};

}
