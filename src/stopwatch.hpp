#pragma once

#include <chrono>
namespace emida
{

class stopwatch
{
public:
	stopwatch() : stopwatch(true) {}

	stopwatch(bool activate)
		: active_(activate)
		, start_(c.now())
		, total_start_(start_)
	{}
	
	void tick(const std::string& label)
	{
		if (!active_)
			return;
		write_duration(label, start_, c.now());
		start_ = c.now();
	}

	void total()
	{
		if (!active_)
			return;
		write_duration("TOTAL:", total_start_, c.now());
	}
		
private:
	std::chrono::high_resolution_clock::time_point start_;
	std::chrono::high_resolution_clock::time_point total_start_;
	std::chrono::high_resolution_clock c;
	bool active_;

	template<typename time_point>
	void write_duration(const std::string & label, time_point start, time_point end)
	{
		auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		std::cout << label << std::to_string(dur.count() / 1000.0) << " ms" << "\n";
	}
};

}