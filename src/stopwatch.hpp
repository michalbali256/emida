
#include <chrono>
#ifdef MEASURE_TIME
#define START_STOPWATCH() \
	std::chrono::high_resolution_clock c; \
	auto start = c.now(); \
	auto total_start = c.now();

#define TICK(label) \
	write_duration(label, start, c.now()); \
	start = c.now();

#define TOTAL() \
	write_duration("TOTAL:", total_start, c.now());
#else
#define START_STOPWATCH()
#define TICK(label)
#define TOTAL()
#endif

namespace emida
{

template<typename time_point>
void write_duration(std::string label, time_point start, time_point end)
{
	auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << label << std::to_string(dur.count() / 1000.0) << " ms" << "\n";
}

}