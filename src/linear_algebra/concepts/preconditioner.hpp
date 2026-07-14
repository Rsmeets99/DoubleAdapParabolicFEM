#pragma once

#include <concepts>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace la::concepts
{
    enum class PreconditionerType
    {
        None,
        ParabolicGraphNorm
    };

    enum class PreconditionerSafetyClass
    {
        NoPreconditioner,
        SpdCompatible,
        Nonsymmetric,
        UnknownUnsafe
    };

    struct PreconditionerDiagnostics
    {
        double setup_seconds = 0.0;
        std::optional<bool> assumes_spd{};
        std::optional<bool> assumes_symmetric{};
        std::optional<bool> is_spd{};
        std::optional<bool> is_symmetric{};
        std::optional<std::string> validation_rejection_reason{};
    };

    struct PreconditionerSafety
    {
        PreconditionerSafetyClass safety_class =
            PreconditionerSafetyClass::UnknownUnsafe;
        bool assumes_spd = false;
        bool assumes_symmetric = false;
        std::string_view description = "unknown or unsafe preconditioner";

        [[nodiscard]] constexpr bool is_none() const noexcept
        {
            return safety_class ==
                PreconditionerSafetyClass::NoPreconditioner;
        }

        [[nodiscard]] constexpr bool is_spd_compatible() const noexcept
        {
            return safety_class ==
                   PreconditionerSafetyClass::SpdCompatible ||
                   safety_class ==
                   PreconditionerSafetyClass::NoPreconditioner;
        }

        [[nodiscard]] constexpr bool is_nonsymmetric() const noexcept
        {
            return safety_class ==
                PreconditionerSafetyClass::Nonsymmetric;
        }
    };

    [[nodiscard]] constexpr std::string_view preconditioner_type_name(
        PreconditionerType preconditioner) noexcept
    {
        switch (preconditioner)
        {
        case PreconditionerType::None:
            return "none";
        case PreconditionerType::ParabolicGraphNorm:
            return "parabolic_graph_norm";
        }

        return "unknown";
    }

    [[nodiscard]] constexpr PreconditionerSafety classify_preconditioner(
        PreconditionerType preconditioner) noexcept
    {
        switch (preconditioner)
        {
        case PreconditionerType::None:
            return {
                PreconditionerSafetyClass::NoPreconditioner,
                true,
                true,
                "identity preconditioner"};
        case PreconditionerType::ParabolicGraphNorm:
            return {
                PreconditionerSafetyClass::SpdCompatible,
                true,
                true,
                "parabolic graph-norm SPD-compatible preconditioner"};
        }

        return {};
    }

    inline void apply_preconditioner_safety_to_diagnostics(
        PreconditionerDiagnostics& diagnostics,
        PreconditionerType preconditioner)
    {
        const PreconditionerSafety safety =
            classify_preconditioner(preconditioner);
        diagnostics.assumes_spd = safety.assumes_spd;
        diagnostics.assumes_symmetric = safety.assumes_symmetric;

        if (safety.is_none())
        {
            diagnostics.is_spd = true;
            diagnostics.is_symmetric = true;
        }
    }

    template<class Preconditioner, class Vector>
    concept PreconditionerApplyLike =
        requires(const Preconditioner& P, const Vector& r)
        {
            P.solve(r);
        };

    template<class Preconditioner, class Matrix>
    concept PreconditionerSetupLike =
        requires(Preconditioner P, const Matrix& A)
        {
            P.compute(A);
        };

    template<class Preconditioner, class Matrix, class Vector>
    concept MatrixPreconditionerLike =
        PreconditionerApplyLike<Preconditioner, Vector> &&
        PreconditionerSetupLike<Preconditioner, Matrix>;
}
