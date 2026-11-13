module;

#include <span>
#include <cmath>
#include <concepts>

export module distance;

export namespace hyperspheres::distance {

    template <typename T>
        requires requires (T x, T y) { {x * y} -> std::convertible_to<T>; {x * x + y * y} -> std::convertible_to<T>; }
    T dot_product_naive(const std::span<T> x, const std::span<T> y) {
        T dot_product {0};
        std::size_t i;
        T xy0, xy1, xy2, xy3, xy4, xy5, xy6, xy7;
        xy0 = xy1 = xy2 = xy3 = xy4 = xy5 = xy6 = xy7 = 0;
        for (i = 0; i + 7 < x.size(); i += 8) {
            xy0 = x[i + 0] * y[i + 0] + xy0;
            xy1 = x[i + 1] * y[i + 1] + xy1;
            xy2 = x[i + 2] * y[i + 2] + xy2;
            xy3 = x[i + 3] * y[i + 3] + xy3;
            xy4 = x[i + 4] * y[i + 4] + xy4;
            xy5 = x[i + 5] * y[i + 5] + xy5;
            xy6 = x[i + 6] * y[i + 6] + xy6;
            xy7 = x[i + 7] * y[i + 7] + xy7;
        }
        dot_product = ((xy0 + xy4) + (xy1 + xy5)) + ((xy2 + xy6) + (xy3 + xy7));
        if (i != x.size()) [[unlikely]] {
            for (;i < x.size(); ++i) {
                dot_product = x[i] * y[i] + dot_product;
            }
        }
        return dot_product;
    }

    template <typename T>
        requires requires (T x, T y) { {x * y} -> std::convertible_to<T>; {x * x + y * y} -> std::convertible_to<T>; }
    void normalize(std::span<T> x) {
        T norm_x = dot_product_naive<T>(x, x);
        if (norm_x == 0) [[unlikely]] {
            // todo: maybe better throw
            return;
        }
        norm_x = std::sqrt(norm_x);
        std::size_t i;
        for (i = 0; i + 7 < x.size(); i += 8) {
            x[i + 0] = x[i + 0] / norm_x;
            x[i + 1] = x[i + 1] / norm_x;
            x[i + 2] = x[i + 2] / norm_x;
            x[i + 3] = x[i + 3] / norm_x;
            x[i + 4] = x[i + 4] / norm_x;
            x[i + 5] = x[i + 5] / norm_x;
            x[i + 6] = x[i + 6] / norm_x;
            x[i + 7] = x[i + 7] / norm_x;
        }
        if (i != x.size()) [[unlikely]] {
            for (;i < x.size(); ++i) {
                x[i] = x[i] / norm_x;
            }
        }
    }

    template <typename T>
        requires requires (T x, T y) { {x * y} -> std::convertible_to<T>; {x * x + y * y} -> std::convertible_to<T>; }
    T squared_normalized_l2_distance(const std::span<T> x, const std::span<T> y) {
        const T dot_product = dot_product_naive<T>(x, y);
        const T dist = T{2} - T{2} * dot_product;
        if (dist <= 0) [[unlikely]] {
            return T{0};
        }
        return dist;
    }

    template <typename T>
        requires requires (T x, T y) { {x * y} -> std::convertible_to<T>; {x * x + y * y} -> std::convertible_to<T>; }
    T squared_l2_distance(const std::span<T> x, const std::span<T> y) {
        T dist {0};
        std::size_t i;
        T xy0, xy1, xy2, xy3, xy4, xy5, xy6, xy7;
        T diff0, diff1, diff2, diff3, diff4, diff5, diff6, diff7;
        xy0 = xy1 = xy2 = xy3 = xy4 = xy5 = xy6 = xy7 = 0;
        diff0 = diff1 = diff2 = diff3 = diff4 = diff5 = diff6 = diff7 = 0;
        for (i = 0; i + 7 < x.size(); i += 8) {
            diff0 = x[i + 0] - y[i + 0];
            diff1 = x[i + 1] - y[i + 1];
            diff2 = x[i + 2] - y[i + 2];
            diff3 = x[i + 3] - y[i + 3];
            diff4 = x[i + 4] - y[i + 4];
            diff5 = x[i + 5] - y[i + 5];
            diff6 = x[i + 6] - y[i + 6];
            diff7 = x[i + 7] - y[i + 7];
            xy0 = diff0 * diff0 + xy0;
            xy1 = diff1 * diff1 + xy1;
            xy2 = diff2 * diff2 + xy2;
            xy3 = diff3 * diff3 + xy3;
            xy4 = diff4 * diff4 + xy4;
            xy5 = diff5 * diff5 + xy5;
            xy6 = diff6 * diff6 + xy6;
            xy7 = diff7 * diff7 + xy7;
        }
        dist = ((xy0 + xy4) + (xy1 + xy5)) + ((xy2 + xy6) + (xy3 + xy7));
        if (i != x.size()) [[unlikely]] {
            for (;i < x.size(); ++i) {
                dist = (x[i] - y[i]) * (x[i] - y[i]) + dist;
            }
        }
        return dist;
    }

    template <typename T>
        requires requires (T x, T y) { {x * y} -> std::convertible_to<T>; {x * x + y * y} -> std::convertible_to<T>; }
    T normalized_l2_distance(const std::span<T> x, const std::span<T> y) {
        const T dist = squared_normalized_l2_distance<T>(x, y);
        return std::sqrt(dist);
    }

    float normalized_l2_distance(const std::span<float> x, const std::span<float> y) {
        return normalized_l2_distance<float>(x, y);
    }

    template <typename T, typename Function>
        requires std::invocable<Function> && requires (T x, T y) { {x * y} -> std::convertible_to<T>; {x * x + y * y} -> std::convertible_to<T>; }
    T calculate_similarity(const std::span<T> x, const std::span<T> y, Function func) {
        return func(x, y);
    }

    template <typename T>
        requires requires (T x, T y) { {x * y} -> std::convertible_to<T>; {x * x + y * y} -> std::convertible_to<T>; }
    T calculate_similarity(const std::span<T> x, const std::span<T> y) {
        return calculate_similarity<std::decay_t<T>, std::decay_t<decltype(normalized_l2_distance<T>)>>(x, y, normalized_l2_distance<T>);
    }

}