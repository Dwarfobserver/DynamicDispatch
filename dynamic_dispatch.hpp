
// Implements optimized dynamic dispatch where the possible states are known
// at compile-time (like for std::variant).

#pragma once

#include <array>
#include <utility>
#include <variant>

namespace dyd {

// +-----------+
// | Interface |
// +-----------+

// __________________
// Dispatch functions

// 'dispatch(value, f)' will call 'f(tag)', where decltype(tag)::value is the
// value available at compile-time.
template <class T, class F>
decltype(auto) dispatch(T&& value, F && f);

// The dispatch on variants are equivalents to a more optimized 'std::visit'.
// It does not handle the state 'valueness_by_exception'.
template <class...Ts, class F>
decltype(auto) dispatch(std::variant<Ts...> & v, F && f);
template <class...Ts, class F>
decltype(auto) dispatch(std::variant<Ts...> const& v, F && f);
template <class...Ts, class F>
decltype(auto) dispatch(std::variant<Ts...> && v, F && f);

// ___________
// Finite sets

// Lists all possible values associated to a given type in a std::array.
template <class T>
struct finite_set;

// Finite set implementation for booleans :
template <>
struct finite_set<bool> {
    static constexpr auto value = std::array{ false, true };
};

// Indicates if a class holds a finite set of values (like for 'T = finite_set<bool>').
template <class T, class SFINAE = void>
constexpr bool is_finite_set_v = false;

// Indicates if a class holds a finite set of values represented as contiguous
// integral values.
template <class T, class SFINAE = void>
constexpr bool is_contiguous_finite_set_v = false;

// Creates an std::array with all values between 'First' and 'Last'.
template <auto First, auto Last>
constexpr auto make_contiguous_array() noexcept;

// Conveniant alias to have a class holding a continuous array.
template <auto First, auto Last>
struct contiguous_array_t {
    static constexpr auto value = make_contiguous_array<First, Last>();
};

// ___________________
// Dispatch strategies

// Algorithms which dispatch the function call.
// Used with Strategy::operator(T&& value, F&& f), to f(tag). 

// Usually the most optimized strategy.
// Expands to :
// | switch (index) {
// |     case FiniteSet[0]: return f(value_tag<FiniteSet[0]>{}); 
// |     ...
// |     case [N-1]: return f(value_tag<[N-1]>{}); 
// | };
template <class FiniteSet>
struct switch_strategy;

// O(1) algorithm. Prevents inlining on most compilers.
// Expands to :
// | // return f(value_tag<index>{});
// | return jump_table[index](f);
template <class FiniteSet>
struct jump_table_strategy;

// O(N) algorithm.
// Expands to :
// | if (index == FiniteSet[0]) return f(value_tag<FiniteSet[0]>{});
// | else if (index == [1]) return f(value_tag<[1]>{});
// | else if ...
// | else return f(value_tag<[N-1]>{});
template <class FiniteSet>
struct sequential_search_strategy;

// O(log(N)) algorithm.
// | if (index < FiniteSet[N/2]) {
// |     if (index < FiniteSet[N/4])      { ... }
// |     else if (index > FiniteSet[N/4]) { ... }
// |     else return f(value_tag<FiniteSet[N/4]>{});
// | }
// | else if (index > FiniteSet[N/2]) {
// |     ...
// | }
// | else return f(value_tag<FiniteSet[N/2]>{});
template <class FiniteSet>
struct binary_search_strategy;

// Indicates if the given strategy is available.
// True for all strategies, except for switch_strategy (where the finite set size is limited).
template <class Strategy>
constexpr bool is_available_strategy_v = true;

// Chooses the switch strategy if she is available, else use the jump table for huge contiguous
// finite sets. Else, use the binary search strategy for huge sets and the sequential search
// for little sets.
template <class FiniteSet>
using default_strategy_t = std::conditional_t<
    is_available_strategy_v<switch_strategy<FiniteSet>>,
    switch_strategy<FiniteSet>,
    std::conditional_t<
        is_contiguous_finite_set_v<FiniteSet> && (FiniteSet::value.size() > 10),
        jump_table_strategy<FiniteSet>,
        std::conditional_t<
            (FiniteSet::value.size() > 5),
            binary_search_strategy<FiniteSet>,
            sequential_search_strategy<FiniteSet>
        >
    >
>;

// +----------------+
// | Implementation |
// +----------------+

// __________
// Finite set

namespace detail {
    template <auto Value>
    struct value_tag {
        static constexpr auto value = Value;
    };

    template <class T>
    constexpr bool is_integer_type_v = std::is_integral_v<T> || std::is_enum_v<T>;
    
    template <class T>
    constexpr auto make_type() {
        if constexpr (std::is_enum_v<T>) {
            return std::underlying_type_t<T>{};
        }
        else {
            return T{};
        }
    }

    template <class T>
    struct integer_type {
        static_assert(is_integer_type_v<T>);

        using type = decltype(make_type<T>());
    };
    template <class T>
    using integer_type_t = typename integer_type<T>::type;

    template <class ValueT, auto IntFirst, size_t...Is>
    constexpr auto make_contiguous_array(std::index_sequence<Is...>) noexcept {
        return std::array{ static_cast<ValueT>(IntFirst + Is) ... };
    }
}

template <auto First, auto Last>
constexpr auto make_contiguous_array() noexcept {
    using value_type = decltype(First);
    static_assert(std::is_same_v<value_type, decltype(Last)>);

    using int_t = detail::integer_type_t<value_type>;
    constexpr auto i_first = static_cast<int_t>(First);
    constexpr auto i_last  = static_cast<int_t>(Last);
    static_assert(i_first <= i_last);

    using sequence_type = std::make_index_sequence<1 + i_last - i_first>;
    return detail::make_contiguous_array<value_type, i_first>(sequence_type{});
}

// Workaround for MSVC
template <class T>
constexpr bool is_finite_set_v<T, std::enable_if_t<
    true//(T::value.size() >= 1)
>> = T::value.size() >= 1;

namespace detail {
    template <class FiniteSet, size_t...Is>
    constexpr bool is_contiguous_finite_set(std::index_sequence<Is...>) {
        constexpr auto first = std::get<0>(FiniteSet::value);

        using int_t = detail::integer_type_t<decltype(first)>;
        constexpr auto i_first = static_cast<int_t>(first);

        return ((i_first == static_cast<int_t>(std::get<Is>(FiniteSet::value)) - static_cast<int_t>(Is)) && ...);
    }
}

template <class T>
constexpr bool is_contiguous_finite_set_v<T, std::enable_if_t<
    is_finite_set_v<T> &&
    detail::is_integer_type_v<decltype(T::value)::template value_type>
>> = detail::is_contiguous_finite_set<T>(std::make_index_sequence<T::value.size()>{});

// __________
// Strategies

#define DYD_PP_FWD(x) std::forward<decltype(x)>(x)

namespace detail {
    template <class F, auto Value>
    constexpr auto make_jump_fn() {
        return +[] (F f) {
            return DYD_PP_FWD(f)(value_tag<Value>{});
        };
    }
    template <class FiniteSet, class F, size_t...Is>
    constexpr auto make_jump_table(std::index_sequence<Is...>) {
        static_assert(std::is_reference_v<F>);
        return std::array{ make_jump_fn<F, std::get<Is>(FiniteSet::value)>()... };
    }
    template <class T>
    using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;
}

template <class FiniteSet>
struct jump_table_strategy {
    static_assert(is_contiguous_finite_set_v<FiniteSet>);

    template <class T, class F>
    decltype(auto) operator()(T&& value, F&& f) const {
        using sequence_type = std::make_index_sequence<FiniteSet::value.size()>;
        constexpr auto jump_table = detail::make_jump_table<FiniteSet, F&&>(sequence_type{});

        using int_t = detail::integer_type_t<detail::remove_cvref_t<T>>;
        constexpr auto offset = static_cast<int_t>(std::get<0>(FiniteSet::value));

        const auto index = static_cast<int_t>(DYD_PP_FWD(value)) - offset;
        return jump_table[index](DYD_PP_FWD(f));
    }
};

namespace detail {
    template <class FiniteSet, size_t Size>
    struct switch_dispatcher : std::false_type {};
}

template <class FiniteSet>
constexpr bool is_available_strategy_v<
    switch_strategy<FiniteSet>
> = !std::is_base_of_v<std::false_type, detail::switch_dispatcher<FiniteSet, FiniteSet::value.size()>>;

namespace detail {
    template <class FiniteSet>
    struct switch_dispatcher<FiniteSet, 1> {
        template <class T, class F>
        decltype(auto) operator()(T&&, F&& f) const {
            return DYD_PP_FWD(f)(value_tag<std::get<0>(FiniteSet::value)>{});
        }
    };
}

template <class FiniteSet>
struct switch_strategy {
    static_assert(is_finite_set_v<FiniteSet>);

    template <class T, class F>
    decltype(auto) operator()(T&& value, F&& f) const {
        static_assert(is_available_strategy_v<switch_strategy>);

        using dispatcher = detail::switch_dispatcher<FiniteSet, FiniteSet::value.size()>;
        return dispatcher{}(DYD_PP_FWD(value), DYD_PP_FWD(f));
    }
};

template <class FiniteSet>
struct sequential_search_strategy {};

template <class FiniteSet>
struct binary_search_strategy {};

// ________
// Dispatch

template <class T, class F>
decltype(auto) dispatch(T&& value, F&& f) {
    using set_type = finite_set<detail::remove_cvref_t<T>>;
    static_assert(is_finite_set_v<set_type>);

    using strategy_type = default_strategy_t<set_type>;
    return strategy_type{}(DYD_PP_FWD(value), DYD_PP_FWD(f));
}

namespace detail {
    template <class RetT, class T>
    constexpr decltype(auto) move_if_rvalue(T&& value) noexcept {
        static_assert(std::is_reference_v<RetT>);
        using value_type = std::remove_reference_t<T>;
        using ref_type   = std::conditional_t<std::is_rvalue_reference_v<RetT>,
                           value_type&&, value_type&>;

        return static_cast<ref_type>(value);
    }

    template <class Variant, class F>
    decltype(auto) dispatch_variant(Variant&& v, F&& f) {
        using variant_type  = std::remove_reference_t<Variant>;
        using set_type      = contiguous_array_t<size_t{0}, std::variant_size_v<variant_type> - 1>;
        using strategy_type = default_strategy_t<set_type>;

        return strategy_type{}(v.index(), [&] (auto tag) {
            constexpr auto index = decltype(tag)::value;
            using value_type = std::variant_alternative_t<index, variant_type>;
            using ref_type   = std::conditional_t<std::is_const_v<variant_type>, value_type const&, value_type&>;
            
            auto&& ref = reinterpret_cast<ref_type>(v);
            return DYD_PP_FWD(f)(move_if_rvalue<Variant&&>(ref));
        });
    }
}

template <class...Ts, class F>
decltype(auto) dispatch(std::variant<Ts...> & v, F && f) {
    return detail::dispatch_variant(v, DYD_PP_FWD(f));
}
template <class...Ts, class F>
decltype(auto) dispatch(std::variant<Ts...> const& v, F && f) {
    return detail::dispatch_variant(v, DYD_PP_FWD(f));
}
template <class...Ts, class F>
decltype(auto) dispatch(std::variant<Ts...> && v, F && f) {
    return detail::dispatch_variant(std::move(v), DYD_PP_FWD(f));
}

// ____________________________
// Automatic switchs generation

#if _MSC_VER
#define DYD_PP_UNREACHABLE() __assume(false)
#define DYD_PP_EXPECT(x, exp) static_cast<bool>(x)
#else
#define DYD_PP_UNREACHABLE() __builtin_unreachable()
#define DYD_PP_EXPECT(x, exp) __builtin_expect(static_cast<bool>(x), exp)
#endif

#define DYD_PP_REP_0(...)
#define DYD_PP_REP_1(...) __VA_ARGS__
#define DYD_PP_REP_2(...) DYD_PP_REP_1(__VA_ARGS__) , __VA_ARGS__
#define DYD_PP_REP_3(...) DYD_PP_REP_2(__VA_ARGS__) , __VA_ARGS__
#define DYD_PP_REP_4(...) DYD_PP_REP_3(__VA_ARGS__) , __VA_ARGS__
#define DYD_PP_REP_5(...) DYD_PP_REP_4(__VA_ARGS__) , __VA_ARGS__
#define DYD_PP_REP_6(...) DYD_PP_REP_5(__VA_ARGS__) , __VA_ARGS__
#define DYD_PP_REP_7(...) DYD_PP_REP_6(__VA_ARGS__) , __VA_ARGS__
#define DYD_PP_REP_8(...) DYD_PP_REP_7(__VA_ARGS__) , __VA_ARGS__
#define DYD_PP_REP_9(...) DYD_PP_REP_8(__VA_ARGS__) , __VA_ARGS__
#define DYD_PP_REP_10(...) DYD_PP_REP_9(__VA_ARGS__) , __VA_ARGS__

#define DYD_PP_REP1(nb, ...) \
  DYD_PP_REP_##nb(__VA_ARGS__)

#define DYD_PP_REP2(tens, ones, ...) \
  DYD_PP_REP_##tens(DYD_PP_REP_10(__VA_ARGS__)) , \
  DYD_PP_REP_##ones(__VA_ARGS__)

#define DYD_PP_REP3(hundreds, tens, ones, ...) \
  DYD_PP_REP_##hundreds(DYD_PP_REP_10(DYD_PP_REP_10(__VA_ARGS__))) , \
  DYD_PP_REP_##tens(DYD_PP_REP_10(__VA_ARGS__)) , \
  DYD_PP_REP_##ones(__VA_ARGS__)

#define DYD_PP_EVAL0(...) __VA_ARGS__
#define DYD_PP_EVAL1(...) DYD_PP_EVAL0 (DYD_PP_EVAL0 (DYD_PP_EVAL0 (__VA_ARGS__)))
#define DYD_PP_EVAL2(...) DYD_PP_EVAL1 (DYD_PP_EVAL1 (DYD_PP_EVAL1 (__VA_ARGS__)))
#define DYD_PP_EVAL3(...) DYD_PP_EVAL2 (DYD_PP_EVAL2 (DYD_PP_EVAL2 (__VA_ARGS__)))
#define DYD_PP_EVAL4(...) DYD_PP_EVAL3 (DYD_PP_EVAL3 (DYD_PP_EVAL3 (__VA_ARGS__)))
#define DYD_PP_EVAL(...)  DYD_PP_EVAL4 (DYD_PP_EVAL4 (DYD_PP_EVAL4 (__VA_ARGS__)))

#define DYD_PP_EMPTY
#define DYD_PP_EMPTY_ARGS(...)

#define DYD_PP_MAP_GET_END() 0, DYD_PP_EMPTY_ARGS

#define DYD_PP_MAP_NEXT0(item, next, ...) next DYD_PP_EMPTY
#if defined(_MSC_VER)
#define DYD_PP_MAP_NEXT1(item, next) DYD_PP_EVAL0(DYD_PP_MAP_NEXT0 (item, next, 0))
#else
#define DYD_PP_MAP_NEXT1(item, next) DYD_PP_MAP_NEXT0 (item, next, 0)
#endif
#define DYD_PP_MAP_NEXT(item, next)  DYD_PP_MAP_NEXT1 (DYD_PP_MAP_GET_END item, next)

#define DYD_PP_MAP0(f, n, x, peek, ...) f(n, x) DYD_PP_MAP_NEXT (peek, DYD_PP_MAP1) (f, n+1, peek, __VA_ARGS__)
#define DYD_PP_MAP1(f, n, x, peek, ...) f(n, x) DYD_PP_MAP_NEXT (peek, DYD_PP_MAP0) (f, n+1, peek, __VA_ARGS__)
#define DYD_PP_MAP(f, ...) DYD_PP_EVAL (DYD_PP_MAP1 (f, 0, __VA_ARGS__, (), 0))

#define DYD_PP_CASE(nb, x) \
    case std::get<nb>(FiniteSet::value): return DYD_PP_FWD(f)(value_tag<std::get<nb>(FiniteSet::value)>{});

#define DYD_MAKE_SWITCH_1(nb) \
namespace dyd::detail { \
    template <class FiniteSet> \
    struct switch_dispatcher<FiniteSet, nb> { \
        template <class T, class F> \
        decltype(auto) operator()(T&& value, F&& f) const { \
            switch (value) { \
                DYD_PP_MAP(DYD_PP_CASE, DYD_PP_REP1(nb, x)) \
            } \
            DYD_PP_UNREACHABLE(); \
        } \
    }; \
} struct dyd_force_semicolon_

#define DYD_MAKE_SWITCH_2(tens, ones) \
namespace dyd::detail { \
    template <class FiniteSet> \
    struct switch_dispatcher<FiniteSet, tens##ones> { \
        template <class T, class F> \
        decltype(auto) operator()(T&& value, F&& f) const { \
            switch (value) { \
                DYD_PP_MAP(DYD_PP_CASE, DYD_PP_REP2(tens, ones, x)) \
            } \
            DYD_PP_UNREACHABLE(); \
        } \
    }; \
} struct dyd_force_semicolon_

#define DYD_MAKE_SWITCH_3(hundreds, tens, ones) \
namespace dyd::detail { \
    template <class FiniteSet> \
    struct switch_dispatcher<FiniteSet, hundreds##tens##ones> { \
        template <class T, class F> \
        decltype(auto) operator()(T&& value, F&& f) const { \
            switch (value) { \
                DYD_PP_MAP(DYD_PP_CASE, DYD_PP_REP3(hundreds, tens, ones, x)) \
            } \
            DYD_PP_UNREACHABLE(); \
        } \
    }; \
} struct dyd_force_semicolon_

} // ::dyd

DYD_MAKE_SWITCH_1(2);
DYD_MAKE_SWITCH_1(3);
DYD_MAKE_SWITCH_1(4);
DYD_MAKE_SWITCH_1(5);
DYD_MAKE_SWITCH_1(6);
DYD_MAKE_SWITCH_1(7);
DYD_MAKE_SWITCH_1(8);
DYD_MAKE_SWITCH_1(9);
DYD_MAKE_SWITCH_2(1,0);
