/* Large Language Model training framework in C++
 * Copyright © 2026 Samuel Andersen
 *  ________   ___   __    ______   ______   ______    ______   ______   ___   __    ______   ________   ___ __ __     
 * /_______/\ /__/\ /__/\ /_____/\ /_____/\ /_____/\  /_____/\ /_____/\ /__/\ /__/\ /_____/\ /_______/\ /__//_//_/\    
 * \::: _  \ \\::\_\\  \ \\:::_ \ \\::::_\/_\:::_ \ \ \::::_\/_\::::_\/_\::\_\\  \ \\::::_\/_\::: _  \ \\::\| \| \ \   
 *  \::(_)  \ \\:. `-\  \ \\:\ \ \ \\:\/___/\\:(_) ) )_\:\/___/\\:\/___/\\:. `-\  \ \\:\/___/\\::(_)  \ \\:.      \ \  
 *   \:: __  \ \\:. _    \ \\:\ \ \ \\::___\/_\: __ `\ \\_::._\:\\::___\/_\:. _    \ \\_::._\:\\:: __  \ \\:.\-/\  \ \ 
 *    \:.\ \  \ \\. \`-\  \ \\:\/.:| |\:\____/\\ \ `\ \ \ /____\:\\:\____/\\. \`-\  \ \ /____\:\\:.\ \  \ \\. \  \  \ \
 *     \__\/\__\/ \__\/ \__\/ \____/_/ \_____\/ \_\/ \_\/ \_____\/ \_____\/ \__\/ \__\/ \_____\/ \__\/\__\/ \__\/ \__\/    
 *                                                                                                               
 * Project: Large Language Model in C++
 * @author : Samuel Andersen
 * @version: 2026-10-19
 *
 * Checkpoint.hpp: Checkpoint utils
 * 
 */

#ifndef CHECKPOINT_HPP
#define CHECKPOINT_HPP

/* Standard dependencies */
#include <cstdint>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

/* Local dependencies */
#include "AbstractEmbedding.hpp"
#include "Layer.hpp"
#include "Log.hpp"
#include "Tensor.hpp"

namespace Checkpoint_NS {

/* Use the AbstractEmbedding interface */
using AbstractEmbedding_NS::AbstractEmbedding;

/* Use the Layer interface */
using Layer_NS::Layer;

/* Use logging */
using Log_NS::log_message;
using Log_NS::Log_Priority;

/* Use Tensor */
using Tensor_NS::Tensor;

/**
 * Checkpoint class, for reading and writing checkpoints from the various layers
 */
template <typename T>
requires std::is_arithmetic_v<T>
class Checkpoint {
/* Private data elements */
};

}; // namespace Checkpoint_NS

#endif
