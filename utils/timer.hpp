#ifndef _TIMER_HPP_
#define _TIMER_HPP_

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>

template<typename T>
class Timer {
public:
	Timer()
	{
		LARGE_INTEGER clock_freq;
		QueryPerformanceFrequency(&clock_freq);
		m_clock_freq = clock_freq.QuadPart;

		reset();
	}
	~Timer() {}

	void reset()
	{
		LARGE_INTEGER start_time;
		QueryPerformanceCounter(&start_time);
		m_start_clock = start_time.QuadPart;
	}

	T check()
	{
		LARGE_INTEGER check_time;
		QueryPerformanceCounter(&check_time);
		return static_cast<T>(check_time.QuadPart-m_start_clock)/static_cast<T>(m_clock_freq);
	}

private:
	__int64 m_start_clock;
	__int64 m_clock_freq;
};

#endif