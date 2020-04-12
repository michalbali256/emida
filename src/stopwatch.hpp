#pragma once

#include <chrono>

#include "common.hpp"

namespace emida
{

struct stats
{
	inline static size_t total_pics = 0;
	inline static size2_t border = { 0, 0 };
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
		tick(label, start_.size() - 1);
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
	
	inline static stats global_stats;
	inline static bool global_activate;

private:
	inline static const std::string total_ = "TOTAL: ";
	
	std::vector<std::chrono::high_resolution_clock::time_point> start_;
	
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
		std::cerr << indentation_[level < 0 ? 0 : level + bonus_indent_] << label << std::to_string(dur.count() / 1000.0) << " ms" << "\n";
	}
};

}