#pragma once

#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace adaptive_algorithm
{
    template<
        class MFunction,
        class EllFunction,
        class InitialValueFunction,
        class ExactSolution = std::monostate,
        class AuxiliaryData = std::monostate>
    struct ProblemData
    {
        using ExactDataType = ExactSolution;
        using AuxiliaryDataType = AuxiliaryData;

        std::string name{};
        std::string description{};

        MFunction M;
        EllFunction ell;
        InitialValueFunction u0;

        std::optional<ExactSolution> exact{};
        std::optional<AuxiliaryData> auxiliary{};

        [[nodiscard]] bool has_exact_data() const noexcept
        {
            return exact.has_value();
        }

        [[nodiscard]] bool has_auxiliary_data() const noexcept
        {
            return auxiliary.has_value();
        }
    };

    template<class MFunction, class EllFunction, class InitialValueFunction>
    [[nodiscard]] ProblemData<
        std::decay_t<MFunction>,
        std::decay_t<EllFunction>,
        std::decay_t<InitialValueFunction>,
        std::monostate,
        std::monostate>
    make_problem_data(
        MFunction&& M,
        EllFunction&& ell,
        InitialValueFunction&& u0,
        std::string name = {},
        std::string description = {})
    {
        return {
            std::move(name),
            std::move(description),
            std::forward<MFunction>(M),
            std::forward<EllFunction>(ell),
            std::forward<InitialValueFunction>(u0),
            std::nullopt,
            std::nullopt
        };
    }

    template<
        class MFunction,
        class EllFunction,
        class InitialValueFunction,
        class ExactSolution>
    [[nodiscard]] ProblemData<
        std::decay_t<MFunction>,
        std::decay_t<EllFunction>,
        std::decay_t<InitialValueFunction>,
        std::decay_t<ExactSolution>,
        std::monostate>
    make_problem_data(
        MFunction&& M,
        EllFunction&& ell,
        InitialValueFunction&& u0,
        std::string name,
        std::string description,
        ExactSolution&& exact_solution)
    {
        return {
            std::move(name),
            std::move(description),
            std::forward<MFunction>(M),
            std::forward<EllFunction>(ell),
            std::forward<InitialValueFunction>(u0),
            std::optional<std::decay_t<ExactSolution>>{
                std::forward<ExactSolution>(exact_solution)},
            std::nullopt
        };
    }

    template<
        class MFunction,
        class EllFunction,
        class InitialValueFunction,
        class ExactSolution,
        class AuxiliaryData>
    [[nodiscard]] ProblemData<
        std::decay_t<MFunction>,
        std::decay_t<EllFunction>,
        std::decay_t<InitialValueFunction>,
        std::decay_t<ExactSolution>,
        std::decay_t<AuxiliaryData>>
    make_problem_data(
        MFunction&& M,
        EllFunction&& ell,
        InitialValueFunction&& u0,
        std::string name,
        std::string description,
        ExactSolution&& exact_solution,
        AuxiliaryData&& auxiliary)
    {
        return {
            std::move(name),
            std::move(description),
            std::forward<MFunction>(M),
            std::forward<EllFunction>(ell),
            std::forward<InitialValueFunction>(u0),
            std::optional<std::decay_t<ExactSolution>>{
                std::forward<ExactSolution>(exact_solution)},
            std::optional<std::decay_t<AuxiliaryData>>{
                std::forward<AuxiliaryData>(auxiliary)}
        };
    }
}
