#pragma once
#include <chrono>

namespace TimerUtil {
	template<typename T = std::chrono::milliseconds>
	class Timer
	{
	public:
		Timer() {}
		~Timer() = default;
		Timer(const Timer& _) = delete;
		Timer& operator=(const Timer& _) = delete;

		void Start() { m_stamp = std::chrono::steady_clock::now(); }
		size_t Elapsed() { return std::chrono::duration_cast<T>(std::chrono::steady_clock::now() - m_stamp).count(); }

	private:
		std::chrono::time_point<std::chrono::steady_clock> m_stamp;
	};
}