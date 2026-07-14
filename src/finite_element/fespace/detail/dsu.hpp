#pragma once

#include <numeric>
#include <utility>
#include <vector>

namespace finite_element::fespace::detail
{
    struct DSU
    {
        std::vector<int> parent;
        std::vector<int> rank;

        explicit DSU(int n)
            : parent(static_cast<std::size_t>(n)), rank(static_cast<std::size_t>(n), 0)
        {
            std::iota(parent.begin(), parent.end(), 0);
        }

        int find(int x)
        {
            if (parent[static_cast<std::size_t>(x)] != x)
                parent[static_cast<std::size_t>(x)] = find(parent[static_cast<std::size_t>(x)]);
            return parent[static_cast<std::size_t>(x)];
        }

        void unite(int a, int b)
        {
            a = find(a);
            b = find(b);
            if (a == b)
                return;

            if (rank[static_cast<std::size_t>(a)] < rank[static_cast<std::size_t>(b)])
                std::swap(a, b);

            parent[static_cast<std::size_t>(b)] = a;
            if (rank[static_cast<std::size_t>(a)] == rank[static_cast<std::size_t>(b)])
                ++rank[static_cast<std::size_t>(a)];
        }
    };
}
