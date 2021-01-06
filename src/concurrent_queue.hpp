#pragma once

#include <queue>
#include <condition_variable>
#include <mutex>

namespace emida
{


// Implementation of a thread-safe queue that can be used to pass work between stage of parallel pipeline.
// Cannot be used with multiple consumers.
template<typename T>
class concurrent_queue
{
private:
	std::queue<T> data;
	mutable std::mutex the_mutex;
	std::condition_variable con_empty;
	std::condition_variable con_full;

	const size_t max_size;

public:

	concurrent_queue(size_t capacity = 1) : max_size(capacity) {}

	void wait_for_data()
	{
		std::unique_lock lock(the_mutex);
		while (data.empty())
		{
			con_empty.wait(lock);
		}
	}

	void push(const T& item)
	{
		std::unique_lock lock(the_mutex);

		con_full.wait(lock, [&]() {return data.size() < max_size; });

		data.push(item);

		lock.unlock();

		con_empty.notify_one();

	}

	bool empty() const
	{
		std::lock_guard lock(the_mutex);
		return data.empty();
	}

	T& front()
	{
		std::lock_guard lock(the_mutex);
		return data.front();
	}

	const T& front() const
	{
		std::lock_guard lock(the_mutex);
		return data.front();
	}

	void pop()
	{
		std::unique_lock lock(the_mutex);
		data.pop();
		lock.unlock();
		con_full.notify_one();
	}


};

}