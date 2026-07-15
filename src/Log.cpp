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

#include "include/Log.hpp"
using Log::log_message;
using Log::Log_Priority;
using Log::get_log_priority;


void Log::log_message(Log_Priority priority, const char* caller, const std::string& message) {

    log_message(priority, caller, message.c_str());
}

void Log::log_message(Log_Priority priority, const char* caller, const char* message) {

    // Setup a buffer and get the current time
    std::array<char, Log::LOG_BUFFER_SIZE> buffer = {0};
    time_t t = time(NULL);

    // Format a time string, storing in the buffer
    strftime(buffer.data(), sizeof(buffer), "%Y-%m-%d %H:%M:%S", localtime(&t));

    if (priority == Log_Priority::ERROR) {
        std::cerr << std::format("{}: [{}] - <{}>: ", buffer, get_log_priority(priority), caller) << message << "\n";
    }
    else {
        std::cout << std::format("{}: [{}] - <{}>: ", buffer, get_log_priority(priority), caller) << message << "\n";
    }
}

void Log::log_message(Log_Priority priority, const std::string& caller, const char* message) {

    log_message(priority, caller.c_str(), message);
}

void Log::log_message(Log_Priority priority, const std::string& caller, const std::string& message) {

    log_message(priority, caller.c_str(), message.c_str());
}

constexpr std::string_view Log::get_log_priority(Log_Priority priority) {
    switch (priority) {
        case Log_Priority::DEBUG: 
            return "DEBUG";
            break;
        case Log_Priority::ERROR:
            return "ERROR";
            break;
        case Log_Priority::INFO:
            return "INFO";
            break;
        default:
            return "WARN";
            break;
    }
}
