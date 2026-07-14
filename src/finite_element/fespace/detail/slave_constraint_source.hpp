#pragma once

namespace finite_element::fespace::detail
{
    struct SlaveConstraintSource
    {
        bool is_slave = false;
        bool is_spatial = false;
        int interface_id = -1;
    };
}
