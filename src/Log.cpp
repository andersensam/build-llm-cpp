/*  ________   ___   __    ______   ______   ______    ______   ______   ___   __    ______   ________   ___ __ __     
 * /_______/\ /__/\ /__/\ /_____/\ /_____/\ /_____/\  /_____/\ /_____/\ /__/\ /__/\ /_____/\ /_______/\ /__//_//_/\    
 * \::: _  \ \\::\_\\  \ \\:::_ \ \\::::_\/_\:::_ \ \ \::::_\/_\::::_\/_\::\_\\  \ \\::::_\/_\::: _  \ \\::\| \| \ \   
 *  \::(_)  \ \\:. `-\  \ \\:\ \ \ \\:\/___/\\:(_) ) )_\:\/___/\\:\/___/\\:. `-\  \ \\:\/___/\\::(_)  \ \\:.      \ \  
 *   \:: __  \ \\:. _    \ \\:\ \ \ \\::___\/_\: __ `\ \\_::._\:\\::___\/_\:. _    \ \\_::._\:\\:: __  \ \\:.\-/\  \ \ 
 *    \:.\ \  \ \\. \`-\  \ \\:\/.:| |\:\____/\\ \ `\ \ \ /____\:\\:\____/\\. \`-\  \ \ /____\:\\:.\ \  \ \\. \  \  \ \
 *     \__\/\__\/ \__\/ \__\/ \____/_/ \_____\/ \_\/ \_\/ \_____\/ \_____\/ \__\/ \__\/ \_____\/ \__\/\__\/ \__\/ \__\/    
 *                                                                                                               
 * Project: Basic Neural Network in C++
 * @author : Samuel Andersen
 * @version: 2025-06-24
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#include "include/Log.hpp"
using namespace Log;

/* Define the log priorities we want to use */
const char* LOG_PRIORITY_MAP[4] = {"INFO", "ERROR", "WARNING", "DEBUG"};

void Log::log_message(Log_Priority priority, const char* caller, const std::string& message) {

    // Setup a buffer and get the current time
    char buffer[100];
    memset(buffer, '\0', 100);
    time_t t = time(NULL);

    // Format a time string, storing in the buffer
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", localtime(&t));

    std::cerr << std::format("{}: [{}] - <{}>: ", buffer, LOG_PRIORITY_MAP[priority], caller) << message << "\n";
}

void Log::log_message(Log_Priority priority, const char* caller, const char* message) {

    // Setup a buffer and get the current time
    char buffer[100];
    memset(buffer, '\0', 100);
    time_t t = time(NULL);

    // Format a time string, storing in the buffer
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", localtime(&t));

    std::cerr << std::format("{}: [{}] - <{}>: ", buffer, LOG_PRIORITY_MAP[priority], caller) << message << "\n";
}