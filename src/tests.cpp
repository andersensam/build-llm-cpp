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
 * @version: 2026-04-14
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#include "include/tests.hpp"

using Log::Log_Priority;
using Log::log_message;
using Tokenizer_NS::Tokenizer;

bool BuildLLM_Tests_NS::run_tokenizer_tests(const Tokenizer& t) {

    // Set the default return value to true
    bool ret = true;
    // Set the name of the callsite
    const std::string callsite = "BuildLLM_Tests_NS::run_tokenizer_tests";

    // Run the tests
    bool res1 = Tokenizer_NS::test_tokenizer(t, "It's the last he painted, you know, Mrs. Gisburn said with pardonable pride.");
    bool res2 = Tokenizer_NS::test_tokenizer(t, "This is yet another test to see how this tokenizer works.");

    if (res1) {
        log_message(Log_Priority::INFO, callsite, "Test 1 passed successfully.");
    }
    else {
        log_message(Log_Priority::ERROR, callsite, "Test 1 failed.");
        ret = false;
    }

    if (res2) {
        log_message(Log_Priority::INFO, callsite, "Test 2 passed successfully.");
    }
    else {
        log_message(Log_Priority::ERROR, callsite, "Test 2 failed.");
        ret = false;
    }

    return ret;

}