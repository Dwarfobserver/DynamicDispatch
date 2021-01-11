
#pragma once

// Implements optimized dynamic dispatch where the possible values are known
// at compile-time (like for std::variant through it's index).

#include <string>
#include <array>
#include <variant>

namespace dyd {

// +-----------+
// | Interface |
// +-----------+

// __________________
// Dispatch functions

// 'dispatch(value, f)' will call 'f(value_tag, [optional] index_tag)', where :
// decltype(value_tag)::value is the value, available at compile-time.
// decltype(index_tag)::value is the value's index in it's set at compile-time.
template <class T, class F>
constexpr decltype(auto) dispatch(T && value, F && f);

// The dispatch on variants are equivalents to a more optimized 'std::visit'.
// It does not handle the state 'valueness_by_exception'.
template <class...Ts, class F>
constexpr decltype(auto) dispatch(std::variant<Ts...> & v, F && f);
template <class...Ts, class F>
constexpr decltype(auto) dispatch(std::variant<Ts...> const& v, F && f);
template <class...Ts, class F>
constexpr decltype(auto) dispatch(std::variant<Ts...> && v, F && f);

// ___________
// Finite sets

// Indicates if a class holds a finite set of values.
// It must contains values in a std::array-like container.
// Elements are assumed to be unique.
template <class T>
struct is_finite_set;
template <class T>
constexpr bool is_finite_set_v = is_finite_set<T>::value;

// Set of values associated to a type by default. Can be specialized.
template <class T>
struct default_finite_set;

// Default finite set implementation for booleans :
template <>
struct default_finite_set<bool> {
    static constexpr auto value = std::array{ false, true };
};

// Creates an std::array with all values between 'First' and 'Last' (included).
template <auto First, auto Last>
constexpr auto make_contiguous_array() noexcept;

// Conveniant alias to have a class holding a contiguous array.
template <auto First, auto Last>
struct contiguous_array_t {
    static constexpr auto value = make_contiguous_array<First, Last>();
};

// Indicates if a class holds a finite set of values represented as contiguous
// integer values.
template <class T>
struct is_contiguous_finite_set;
template <class T>
constexpr bool is_contiguous_finite_set_v = is_contiguous_finite_set<T>::value;

// Indicates if a class holds a finite set of values ordered by increasing order.
template <class T>
struct is_ordered_finite_set;
template <class T>
constexpr bool is_ordered_finite_set_v = is_ordered_finite_set<T>::value;

// Indicates if a value can be equivalent to an integral value.
// True for enumerations and integral types.
template <class T>
struct has_integral_representation;
template <class T>
constexpr bool has_integral_representation_v = has_integral_representation<T>::value;

// The function mapping a value to it's integral value. Can be overloaded.
template <class T>
constexpr auto get_integral_representation(T value) noexcept;

// The integral type representing a value (void if not available).
template <class T>
struct integral_representation;
template <class T>
using integral_representation_t = typename integral_representation<T>::type;

// ___________________
// Dispatch strategies

// Algorithms which dispatch the function call.
// Used with Strategy::operator(T&& value, F&& f), to f(tag). 

// Usually the most optimized strategy.
// Requires an integer finite set and a dedicated implementation (which
// creates the switch with the appropriate number of cases).
// Expands to :
// | switch (index) {
// |     case FiniteSet[0]: return f(value_tag<FiniteSet[0]>{}); 
// |     ...
// |     case [N-1]: return f(value_tag<[N-1]>{}); 
// | };
template <class FiniteSet>
struct switch_strategy;

// Generates a switch that handle the given number (until 999) of values in
// the finite set the possible values.
// Digits are separated by commas.
// Must be used in the global namespace.
// Workaround : multiples of ten must pas 'ten' separately.
// Exemple : DYD_MAKE_SWITCH(4,2) generates a switch for sets of 42 values.
// #define DYD_MAKE_SWITCH(digits...)

// O(1) algorithm. Prevents inlining on most compilers.
// Requires a contiguous finite set.
// Expands to :
// | // return f(value_tag<index>{});
// | return jump_table[index](f);
template <class FiniteSet>
struct jump_table_strategy;

// O(N) algorithm.
// Requires the operator '==' to be implemented by the set's values.
// Expands to :
// | if (index == FiniteSet[0]) return f(value_tag<FiniteSet[0]>{});
// | else if (index == [1]) return f(value_tag<[1]>{});
// | else if ...
// | else return f(value_tag<[N-1]>{});
template <class FiniteSet>
struct sequential_search_strategy;

// O(log(N)) algorithm.
// Requires the operator '<' to be implemented by the set's values.
// Expands to :
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

// Indicates if the given strategy is available with it's finite set.
template <class Strategy>
struct is_available_strategy;
template <class Strategy>
constexpr bool is_available_strategy_v = is_available_strategy<Strategy>::value;

// Selects a strategy in this order :
//  - Takes the switch strategy (if she is available, else)
//  - Takes the jump table strategy for sets of > 10 elements (if she is available, else)
//  - Takes the binary search strategy for sets of > 5 elements (if she is available, else)
//  - Takes the sequential search strategy (if she is available, else)
//  - Resolves to 'void'
template <class FiniteSet>
struct default_strategy;
template <class FiniteSet>
using default_strategy_t = typename default_strategy<FiniteSet>::type;

// +----------------+
// | Implementation |
// +----------------+

// __________
// Finite set

// is_finite_set

namespace detail {
    template <class T, class SFINAE = void>
    struct detect_finite_set : std::false_type {};

    template <class T>
    struct detect_finite_set<T, std::enable_if_t<
        std::size(T::value) >= 1
    >> : std::true_type {};
}
template <class T>
struct is_finite_set : detail::detect_finite_set<T> {};

// get_integral_representation

template <class T>
constexpr auto get_integral_representation(T value) noexcept {
    if constexpr (std::is_enum_v<T>) {
        return static_cast<std::underlying_type_t<T>>(value);
    }
    else if constexpr (std::is_integral_v<T>) {
        return value;
    }
}

// (has_)integral_representation

template <class T>
struct integral_representation {
    using type = decltype(get_integral_representation(std::declval<T>()));
};
template <class T>
struct has_integral_representation {
    static constexpr auto value = !std::is_same_v<void, integral_representation_t<T>>;
};

// make_contiguous_array

namespace detail {
    template <class ValueT, auto IntFirst, size_t...Is>
    constexpr auto make_contiguous_array(std::index_sequence<Is...>) noexcept {
        return std::array{ static_cast<ValueT>(IntFirst + Is) ... };
    }
}
template <auto First, auto Last>
constexpr auto make_contiguous_array() noexcept {
    static_assert(std::is_same_v<decltype(First), decltype(Last)>);
    using value_type = decltype(First);

    using int_t = integral_representation_t<value_type>;
    constexpr auto i_first = static_cast<int_t>(First);
    constexpr auto i_last  = static_cast<int_t>(Last);
    static_assert(i_first <= i_last);

    using sequence_type = std::make_index_sequence<1 + i_last - i_first>;
    return detail::make_contiguous_array<value_type, i_first>(sequence_type{});
}

// is_contiguous_finite_set

namespace detail {
    template <class T>
    using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;
    
    template <class T, class SFINAE = void>
    struct detect_contiguous_finite_set : std::false_type {};

    template <class T>
    struct detect_contiguous_finite_set<T, std::enable_if_t<
        is_finite_set_v<T> && has_integral_representation_v<remove_cvref_t<decltype(std::get<0>(T::value))>>
    >> {
        template <size_t...Is>
        static constexpr bool contiguous_values(std::index_sequence<Is...>) {
            constexpr auto first = std::get<0>(T::value);

            using int_t = integral_representation_t<decltype(first)>;

            // set[0] == set[i] - i
            constexpr auto i_first = static_cast<int_t>(first);
            return ((i_first == static_cast<int_t>(std::get<Is>(T::value)) - static_cast<int_t>(Is)) && ...);
        }
        static constexpr bool value = contiguous_values(std::make_index_sequence<std::size(T::value)>{});
    };
}
template <class T>
struct is_contiguous_finite_set : detail::detect_contiguous_finite_set<T> {};

// is_ordered_finite_set

namespace detail {
    template <class T, class SFINAE = void>
    struct detect_ordered_finite_set : std::false_type {};

    template <class T>
    struct detect_ordered_finite_set<T, std::enable_if_t<
        is_finite_set_v<T> // TODO && is detected T < T
    >> {
        template <size_t...Is>
        static constexpr bool contiguous_values(std::index_sequence<Is...>) {
            // set[i] < set[i + 1]
            return ((std::get<Is>(T::value) < std::get<Is + 1>(T::value)) && ...);
        }
        using sequence = std::make_index_sequence<std::size(T::value) - 1>;
        static constexpr bool value = contiguous_values(sequence{});
    };
}
template <class T>
struct is_ordered_finite_set : detail::detect_ordered_finite_set<T> {};

// __________
// Strategies

// value_tag : tag passed to the caller to retrieve the value at compile-time.

namespace detail {
    template <class FiniteSet, size_t I>
    struct value_tag {
        static constexpr auto value = std::get<I>(FiniteSet::value);
    };
}

// is_detected

namespace detail {
    // Workaround vs Clang's std::void_t (which does not discard wrong expressions).
    template <class...>
    struct always_void {
        using type = void;
    };
    template <class...Ts>
    using always_void_t = typename always_void<Ts...>::type;

    template <template <class...> class Expr, class SFINAE, class...Args>
    constexpr bool is_detected_impl = false;
    template <template <class...> class Expr, class...Args>
    constexpr bool is_detected_impl<Expr, always_void_t<Expr<Args...>>, Args...> = true;

    template <template <class...> class Expr, class...Args>
    constexpr bool is_detected_v = is_detected_impl<Expr, void, Args...>;
}

#define DYD_PP_FWD(x) std::forward<decltype(x)>(x)

// call_with_tags : picks between f(value_tag, index_tag) and f(value_tag)

namespace detail {
    template <class FiniteSet, class IntTag, class F>
    using take_index_expr = decltype(
        std::declval<F>()(value_tag<FiniteSet, IntTag::value>{}, IntTag{})
    );

    template <class FiniteSet, size_t I, class F>
    constexpr decltype(auto) call_with_tags(F&& f) {
        using int_tag = std::integral_constant<size_t, I>;
        if constexpr (is_detected_v<take_index_expr, FiniteSet, int_tag, F&&>) {
            return DYD_PP_FWD(f)(value_tag<FiniteSet, I>{}, int_tag{});
        }
        else {
            return DYD_PP_FWD(f)(value_tag<FiniteSet, I>{});
        }
    }
}

// jump_table_strategy

namespace detail {
    template <class F, class FiniteSet, size_t I>
    constexpr auto make_jump_fn() {
        return +[] (F&& f) {
            return call_with_tags<FiniteSet, I>(DYD_PP_FWD(f));
        };
    }
    template <class FiniteSet, class F, size_t...Is>
    constexpr auto make_jump_table(std::index_sequence<Is...>) {
        return std::array{ make_jump_fn<F, FiniteSet, Is>()... };
    }
}
template <class FiniteSet>
struct jump_table_strategy {
    static constexpr auto value = is_contiguous_finite_set_v<FiniteSet>;

    template <class T, class F>
    constexpr decltype(auto) operator()(T value, F && f) const {
        using sequence_type = std::make_index_sequence<FiniteSet::value.size()>;
        constexpr auto jump_table = detail::make_jump_table<FiniteSet, F>(sequence_type{});

        using int_t = integral_representation_t<T>;
        constexpr auto offset = static_cast<int_t>(std::get<0>(FiniteSet::value));

        const auto index = static_cast<int_t>(value) - offset;
        return jump_table[index](DYD_PP_FWD(f));
    }
};

// switch_strategy

namespace detail {
    template <class FiniteSet, size_t Size>
    struct switch_dispatcher : std::false_type {};
    
    template <class FiniteSet, class SFINAE = void>
    struct switch_strategy_impl : std::false_type {};

    template <class FiniteSet>
    struct switch_strategy_impl<FiniteSet, std::enable_if_t<
        is_finite_set_v<FiniteSet>
        && !std::is_base_of_v<std::false_type, switch_dispatcher<FiniteSet, std::size(FiniteSet::value)>>
        && has_integral_representation_v<remove_cvref_t<decltype(std::get<0>(FiniteSet::value))>>
    >> : std::true_type {
        template <class T, class F>
        constexpr decltype(auto) operator()(T value, F && f) const {
            using dispatcher = detail::switch_dispatcher<FiniteSet, std::size(FiniteSet::value)>;
            return dispatcher{}(value, DYD_PP_FWD(f));
        }
    };
}
template <class FiniteSet>
struct switch_strategy : detail::switch_strategy_impl<FiniteSet> {};

// sequential_search_strategy
// TODO : is_detected<get<0>(set) == get<0>(set)>

#if _MSC_VER
#define DYD_PP_UNREACHABLE() __assume(false)
#define DYD_PP_EXPECT(x, exp) static_cast<bool>(x)
#else
#define DYD_PP_UNREACHABLE() __builtin_unreachable()
#define DYD_PP_EXPECT(x, exp) __builtin_expect(static_cast<bool>(x), exp)
#endif

template <class FiniteSet>
struct sequential_search_strategy {
    static constexpr auto value = is_finite_set_v<FiniteSet>;

    template <size_t I>
    struct int_tag {};

    template <class Ret, class T, class F>
    static constexpr Ret try_call(T && value, F && f, int_tag<std::size(FiniteSet::value)>) {
        DYD_PP_UNREACHABLE();
    }
    template <class Ret, class T, class F, size_t I>
    static constexpr Ret try_call(T && value, F && f, int_tag<I>) {
        if (value == std::get<I>(FiniteSet::value)) {
            return detail::call_with_tags<FiniteSet, I>(DYD_PP_FWD(f));
        }
        else {
            return try_call<Ret>(DYD_PP_FWD(value), DYD_PP_FWD(f), int_tag<I + 1>{});
        }
    }

    template <class T, class F>
    constexpr decltype(auto) operator()(T && value, F && f) const {
        using return_type = decltype(detail::call_with_tags<FiniteSet, 0>(DYD_PP_FWD(f)));
        return try_call<return_type>(DYD_PP_FWD(value), DYD_PP_FWD(f), int_tag<0>{});
    }
};

// binary_search_strategy
// TODO : is_detected<get<0>(set) < get<0>(set)>

template <class FiniteSet>
struct binary_search_strategy {
    static constexpr auto value = is_ordered_finite_set_v<FiniteSet>;

    template <size_t...Is>
    struct range_tag {};

    template <class Ret, class T, class F, size_t I>
    static constexpr Ret try_call(T && value, F && f, range_tag<I, 0>) {
        DYD_PP_UNREACHABLE();
    }
    template <class Ret, class T, class F, size_t Begin, size_t Size>
    static constexpr Ret try_call(T && value, F && f, range_tag<Begin, Size>) {

        constexpr auto middle_index = Begin + Size / 2;
        constexpr auto middle_value = std::get<Begin + Size / 2>(FiniteSet::value);

        if (value < middle_value) {
            return try_call<Ret>(DYD_PP_FWD(value), DYD_PP_FWD(f), range_tag<Begin, Size / 2>{});
        }
        else if (middle_value < value) {
            return try_call<Ret>(DYD_PP_FWD(value), DYD_PP_FWD(f), range_tag<middle_index + 1, (Size - 1) / 2>{});
        }
        else {
            return detail::call_with_tags<FiniteSet, middle_index>(DYD_PP_FWD(f));
        }
    }

    template <class T, class F>
    constexpr decltype(auto) operator()(T && value, F && f) const {
        using return_type = decltype(detail::call_with_tags<FiniteSet, 0>(DYD_PP_FWD(f)));
        using range_tag   = range_tag<0, std::size(FiniteSet::value)>;
        return try_call<return_type>(DYD_PP_FWD(value), DYD_PP_FWD(f), range_tag{});
    }
};

// is_available_strategy

template <class Strategy>
struct is_available_strategy {
    static constexpr auto value = Strategy::value;
};

// default_strategy

template <class FiniteSet>
struct default_strategy {
    using type =
    
    std::conditional_t<
        is_available_strategy_v<switch_strategy<FiniteSet>>,
        switch_strategy<FiniteSet>,

    std::conditional_t<
        is_available_strategy_v<jump_table_strategy<FiniteSet>> && (std::size(FiniteSet::value) > 10),
        jump_table_strategy<FiniteSet>,
    
    std::conditional_t<
        is_available_strategy_v<binary_search_strategy<FiniteSet>> && (std::size(FiniteSet::value) > 5),
        binary_search_strategy<FiniteSet>,

    std::conditional_t<
        is_available_strategy_v<sequential_search_strategy<FiniteSet>>,
        sequential_search_strategy<FiniteSet>,
        
    void>>>>;
};

// ________
// Dispatch

template <class T, class F>
constexpr decltype(auto) dispatch(T && value, F && f) {
    using set_type = default_finite_set<detail::remove_cvref_t<T>>;
    static_assert(is_finite_set_v<set_type>);

    using strategy_type = default_strategy_t<set_type>;
    return strategy_type{}(DYD_PP_FWD(value), DYD_PP_FWD(f));
}

namespace detail {
    template <class RetT, class T>
    constexpr decltype(auto) move_if_rvalue(T && value) noexcept {
        static_assert(std::is_reference_v<RetT>);
        using value_type = std::remove_reference_t<T>;
        using ref_type   = std::conditional_t<std::is_rvalue_reference_v<RetT>,
                           value_type &&, value_type &>;

        return static_cast<ref_type>(value);
    }

    template <class Variant, class F>
    constexpr decltype(auto) dispatch_variant(Variant && v, F && f) {
        using variant_type  = std::remove_reference_t<Variant>;
        using set_type      = contiguous_array_t<size_t{0}, std::variant_size_v<variant_type> - 1>;
        using strategy_type = default_strategy_t<set_type>;

        static_assert(is_contiguous_finite_set_v<set_type>);

        return strategy_type{}(v.index(), [&] (auto tag) {
            constexpr auto index = decltype(tag)::value;
            using value_type = std::variant_alternative_t<index, variant_type>;
            using ref_type   = std::conditional_t<std::is_const_v<variant_type>, value_type const&, value_type &>;
            
            auto & ref = reinterpret_cast<ref_type>(v);
            return DYD_PP_FWD(f)(move_if_rvalue<Variant &&>(ref));
        });
    }
}

template <class...Ts, class F>
constexpr decltype(auto) dispatch(std::variant<Ts...> & v, F && f) {
    return detail::dispatch_variant(v, DYD_PP_FWD(f));
}
template <class...Ts, class F>
constexpr decltype(auto) dispatch(std::variant<Ts...> const& v, F && f) {
    return detail::dispatch_variant(v, DYD_PP_FWD(f));
}
template <class...Ts, class F>
constexpr decltype(auto) dispatch(std::variant<Ts...> && v, F && f) {
    return detail::dispatch_variant(std::move(v), DYD_PP_FWD(f));
}

// ____________________________
// Automatic switchs generation

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
  DYD_PP_REP_##tens(DYD_PP_REP_10(__VA_ARGS__)), \
  DYD_PP_REP_##ones(__VA_ARGS__)

#define DYD_PP_REP3(hundreds, tens, ones, ...) \
  DYD_PP_REP_##hundreds(DYD_PP_REP_10(DYD_PP_REP_10(__VA_ARGS__))), \
  DYD_PP_REP_##tens(DYD_PP_REP_10(__VA_ARGS__)), \
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
    case std::get<nb>(FiniteSet::value): return call_with_tags<FiniteSet, nb>(DYD_PP_FWD(f));

#define DYD_PP_SWITCH_BEGIN(nb) \
    namespace dyd::detail { \
        template <class FiniteSet> \
        struct switch_dispatcher<FiniteSet, nb> { \
            template <class T, class F> \
            decltype(auto) operator()(T && value, F && f) const { \
                switch (value) {

#define DYD_PP_SWITCH_END() \
            } \
            DYD_PP_UNREACHABLE(); \
        } \
    }; \
} struct dyd_force_semicolon_

#define DYD_PP_MAKE_SWITCH_1(nb) \
    DYD_PP_SWITCH_BEGIN(nb) \
    DYD_PP_MAP(DYD_PP_CASE, DYD_PP_REP1(nb, x)) \
    DYD_PP_SWITCH_END()

#define DYD_PP_MAKE_SWITCH_2(tens, ones) \
    DYD_PP_SWITCH_BEGIN(10 * tens + ones) \
    DYD_PP_MAP(DYD_PP_CASE, DYD_PP_REP2(tens, ones, x)) \
    DYD_PP_SWITCH_END()

#define DYD_PP_MAKE_SWITCH_3(hundreds, tens, ones) \
    DYD_PP_SWITCH_BEGIN(100 * hundreds + 10 * tens + ones) \
    DYD_PP_MAP(DYD_PP_CASE, DYD_PP_REP3(hundreds, tens, ones, x)) \
    DYD_PP_SWITCH_END()

#define DYD_PP_GET_SWITCH(_1, _2, _3, fn, ...) fn

// Generates a switch that handle the given number (until 999) of
// values in the finite set the possible values.
// Digits are separated by commas.
#define DYD_MAKE_SWITCH(...) DYD_PP_GET_SWITCH( \
    __VA_ARGS__, \
    DYD_PP_MAKE_SWITCH_3, \
    DYD_PP_MAKE_SWITCH_2, \
    DYD_PP_MAKE_SWITCH_1)(__VA_ARGS__)

} // ::dyd

DYD_MAKE_SWITCH(1);
DYD_MAKE_SWITCH(2);
DYD_MAKE_SWITCH(3);
DYD_MAKE_SWITCH(4);
DYD_MAKE_SWITCH(5);
DYD_MAKE_SWITCH(6);
DYD_MAKE_SWITCH(7);
DYD_MAKE_SWITCH(8);
DYD_MAKE_SWITCH(9);
DYD_MAKE_SWITCH(10);
DYD_MAKE_SWITCH(1,1);
