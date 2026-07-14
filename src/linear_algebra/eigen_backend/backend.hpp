#pragma once

#include "backend_types.hpp"

// Including the public Eigen backend header also makes the Eigen
// saddle-preconditioner dispatch specialization visible.  The finite-element
// layer includes only backend-agnostic dispatch declarations.
#include "preconditioners/solve_saddle_system.hpp"
