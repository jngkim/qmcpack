#include <iostream>
#include <chrono>
#include <iomanip>
#include <ctime>

int main() {


// // Inside your function or where you want to print the timestamp
auto now = std::chrono::system_clock::now(); // Get current time point
std::time_t now_c = std::chrono::system_clock::to_time_t(now); // Convert to time_t
std::tm now_tm = *std::localtime(&now_c); // Convert to tm struct for local timezone

// // Print the timestamp
// std::cout << "Current Timestamp: " << std::put_time(&now_tm, "%Y-%m-%d %X") << std::endl;

 // Reset hours, minutes, seconds, and milliseconds to 0 to represent the start of today
 //std::tm now_tm = {0,0,0};
 //   now_tm.tm_hour = 0;
 //   now_tm.tm_min = 0;
    //now_tm.tm_sec = 0;

    auto start_of_today_c = std::mktime(&now_tm); // Convert back to time_t
    auto start_of_today = std::chrono::system_clock::from_time_t(start_of_today_c); // Convert back to time point

    // Calculate duration since start of today
    auto duration_since_start_of_today = now - start_of_today;

    // Convert duration to desired units, e.g., seconds
    auto seconds_since_start_of_today = std::chrono::duration_cast<std::chrono::milliseconds>(duration_since_start_of_today).count();

    // Print the duration since the start of today in seconds
    std::cout << "Seconds since start of today: " << seconds_since_start_of_today << std::endl;


// for(int i=0; i<100; ++i)
// {
// auto time = std::chrono::system_clock::now();
// std::cout << "File name: " << std::to_string(time.time_since_epoch().count()) << std::endl;
// }
    return 0;
}