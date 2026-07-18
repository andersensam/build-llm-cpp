/*  ________   ___   __    ______   ______   ______    ______   ______   ___   __    ______   ________   ___ __ __     
 * /_______/\ /__/\ /__/\ /_____/\ /_____/\ /_____/\  /_____/\ /_____/\ /__/\ /__/\ /_____/\ /_______/\ /__//_//_/\    
 * \::: _  \ \\::\_\\  \ \\:::_ \ \\::::_\/_\:::_ \ \ \::::_\/_\::::_\/_\::\_\\  \ \\::::_\/_\::: _  \ \\::\| \| \ \   
 *  \::(_)  \ \\:. `-\  \ \\:\ \ \ \\:\/___/\\:(_) ) )_\:\/___/\\:\/___/\\:. `-\  \ \\:\/___/\\::(_)  \ \\:.      \ \  
 *   \:: __  \ \\:. _    \ \\:\ \ \ \\::___\/_\: __ `\ \\_::._\:\\::___\/_\:. _    \ \\_::._\:\\:: __  \ \\:.\-/\  \ \ 
 *    \:.\ \  \ \\. \`-\  \ \\:\/.:| |\:\____/\\ \ `\ \ \ /____\:\\:\____/\\. \`-\  \ \ /____\:\\:.\ \  \ \\. \  \  \ \
 *     \__\/\__\/ \__\/ \__\/ \____/_/ \_____\/ \_\/ \_\/ \_____\/ \_____\/ \__\/ \__\/ \_____\/ \__\/\__\/ \__\/ \__\/    
 *                                                                                                               
 * Project: Large Language Model in C++
 * @author : Samuel Andersen
 * @version: 2026-07-14
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#ifndef LOG_HPP
#define LOG_HPP

/* Standard dependencies */
#include <iostream>
#include <cstring>
#include <ctime>
#include <string_view>
#include <array>

namespace Log {

inline constexpr size_t LOG_BUFFER_SIZE = 100;

enum class Log_Priority : uint8_t {
    INFO = 0,
    ERROR = 1,
    WARNING = 2,
    DEBUG = 3
};

/**
 * Log a message to the console
 * @param priority Priority of the message
 * @param caller Name of the caller to the log
 * @param message Message to actually output
 */
void log_message(Log_Priority priority, const char* caller, const std::string& message);

/**
 * Log a message to the console
 * @param priority Priority of the message
 * @param caller Name of the caller to the log
 * @param message Message to actually output
 */
void log_message(Log_Priority priority, const char* caller, const char* message);

/**
 * Log a message to the console
 * @param priority Priority of the message
 * @param caller Name of the caller to the log
 * @param message Message to actually output
 */
void log_message(Log_Priority priority, const std::string& caller, const std::string& message);

/**
 * Log a message to the console
 * @param priority Priority of the message
 * @param caller Name of the caller to the log
 * @param message Message to actually output
 */
void log_message(Log_Priority priority, const std::string& caller, const char* message);

/**
 * Map a Log_Priority to a string
 * @param priority Priority of the message
 * @returns Returns a std::string_view of the Log Priority
 */
constexpr std::string_view get_log_priority(Log_Priority priority);

}; // namespace Log

#endif
