import math
import os
import random
from typing import List, Tuple


def set_seed(seed: int = 42) -> None:
    random.seed(seed)


def vector_dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def vector_add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def vector_sub(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def vector_scalar(a: List[float], s: float) -> List[float]:
    return [x * s for x in a]


def vector_mul(a: List[float], b: List[float]) -> List[float]:
    return [x * y for x, y in zip(a, b)]


def mat_transpose(a: List[List[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*a)]


def mat_mul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    result = [[0.0 for _ in range(len(b[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for k in range(len(b)):
            for j in range(len(b[0])):
                result[i][j] += a[i][k] * b[k][j]
    return result


def mat_vec_mul(a: List[List[float]], v: List[float]) -> List[float]:
    return [vector_dot(row, v) for row in a]


def mat_identity(n: int) -> List[List[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def mat_add(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def mat_scalar(a: List[List[float]], s: float) -> List[List[float]]:
    return [[x * s for x in row] for row in a]


def outer(a: List[float], b: List[float]) -> List[List[float]]:
    return [[x * y for y in b] for x in a]


def mat_inv(a: List[List[float]]) -> List[List[float]]:
    n = len(a)
    aug = [row[:] + identity_row[:] for row, identity_row in zip(a, mat_identity(n))]

    for i in range(n):
        pivot = aug[i][i]
        if abs(pivot) < 1e-10:
            for j in range(i + 1, n):
                if abs(aug[j][i]) > 1e-10:
                    aug[i], aug[j] = aug[j], aug[i]
                    pivot = aug[i][i]
                    break
        if abs(pivot) < 1e-10:
            raise ValueError("Singular matrix")

        scale = 1.0 / pivot
        aug[i] = [val * scale for val in aug[i]]

        for j in range(n):
            if j == i:
                continue
            factor = aug[j][i]
            aug[j] = [aj - factor * ai for aj, ai in zip(aug[j], aug[i])]

    return [row[n:] for row in aug]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def relu(x: float) -> float:
    return x if x > 0 else 0.0


def generate_x(d_c: int, n: int) -> List[List[float]]:
    return [[random.random() for _ in range(d_c)] for _ in range(n)]


def generate_t(t_combo: List[List[float]], t_dist: List[float], n: int) -> List[List[float]]:
    choices = random.choices(t_combo, weights=t_dist, k=n)
    return [choice[:] for choice in choices]


def generate_y_true(coef: List[List[float]], c: float, d: float, x: List[List[float]], t: List[List[float]]) -> List[float]:
    y = []
    for x_i, t_i in zip(x, t):
        xcoef = mat_vec_mul(mat_transpose(coef), x_i)
        u = vector_dot(xcoef, t_i)
        y_error = 0.05 * random.uniform(-1, 1)
        y.append(c / (1 + math.exp(-u)) + d + y_error)
    return y


def generate_y_true_1(coef: List[List[float]], c: float, d: float, x: List[List[float]], t: List[float]) -> List[float]:
    y = []
    for x_i in x:
        xcoef = mat_vec_mul(mat_transpose(coef), x_i)
        u = vector_dot(xcoef, t)
        y.append(c / (1 + math.exp(-u)) + d)
    return y


def build_t_combos(m: int) -> Tuple[List[List[float]], List[float], List[List[float]], List[float]]:
    all_combo = []
    for i in range(1 << m):
        combo = [j + 1 for j in range(m) if i & (1 << j)]
        all_combo.append(combo)

    t_combo = []
    for combo in all_combo:
        t = [0.0 for _ in range(m + 1)]
        t[0] = 1.0
        for idx in combo:
            t[idx] = 1.0
        t_combo.append(t)
    t_dist = [1 / 2**m for _ in range(2**m)]

    t_combo_obs = []
    for i in range(m + 1):
        t = [0.0 for _ in range(m + 1)]
        t[0] = 1.0
        t[i] = 1.0
        t_combo_obs.append(t)
    t = [0.0 for _ in range(m + 1)]
    t[0] = 1.0
    t[1] = 1.0
    t[2] = 1.0
    t_combo_obs.append(t)
    t_dist_obs = [1 / (m + 2) for _ in range(m + 2)]

    return t_combo, t_dist, t_combo_obs, t_dist_obs


class SimpleFNN:
    def __init__(self, d_c: int, m: int, hidden: int = 10) -> None:
        self.d_c = d_c
        self.m = m
        self.hidden = hidden
        self.w1 = [[random.uniform(-0.5, 0.5) for _ in range(hidden)] for _ in range(d_c)]
        self.w2 = [[random.uniform(-0.5, 0.5) for _ in range(m + 1)] for _ in range(hidden)]
        self.c = 1.0

    def forward(self, x: List[float], t: List[float]) -> Tuple[float, List[float], List[float], float, float]:
        h = []
        for j in range(self.hidden):
            s = 0.0
            for k in range(self.d_c):
                s += x[k] * self.w1[k][j]
            h.append(relu(s))

        b = []
        for i in range(self.m + 1):
            s = 0.0
            for j in range(self.hidden):
                s += h[j] * self.w2[j][i]
            b.append(s)

        u = vector_dot(b, t)
        sig = sigmoid(u)
        y_hat = self.c * sig
        return y_hat, b, h, u, sig

    def train_epoch(self, x_train: List[List[float]], t_train: List[List[float]], y_train: List[float], lr: float) -> float:
        total_loss = 0.0
        for x_i, t_i, y_i in zip(x_train, t_train, y_train):
            y_hat, b, h, u, sig = self.forward(x_i, t_i)
            error = y_hat - y_i
            total_loss += error**2

            grad_c = error * sig
            grad_u = error * self.c * sig * (1 - sig)

            grad_b = [grad_u * t_j for t_j in t_i]
            grad_w2 = [[h_j * grad_b_i for grad_b_i in grad_b] for h_j in h]

            grad_h = []
            for j in range(self.hidden):
                grad = 0.0
                for i in range(self.m + 1):
                    grad += grad_b[i] * self.w2[j][i]
                grad_h.append(grad if h[j] > 0 else 0.0)

            grad_w1 = [[x_k * grad_h_j for grad_h_j in grad_h] for x_k in x_i]

            self.c -= lr * grad_c
            for j in range(self.hidden):
                for i in range(self.m + 1):
                    self.w2[j][i] -= lr * grad_w2[j][i]
            for k in range(self.d_c):
                for j in range(self.hidden):
                    self.w1[k][j] -= lr * grad_w1[k][j]

        return total_loss / len(x_train)

    def predict(self, x: List[List[float]], t: List[List[float]]) -> List[float]:
        preds = []
        for x_i, t_i in zip(x, t):
            y_hat, _, _, _, _ = self.forward(x_i, t_i)
            preds.append(y_hat)
        return preds

    def beta(self, x: List[float]) -> List[float]:
        _, b, _, _, _ = self.forward(x, [0.0 for _ in range(self.m + 1)])
        return b


def debiased_prediction(
    net: SimpleFNN,
    x_est: List[List[float]],
    t_est: List[List[float]],
    y_est: List[float],
    t_target: List[float],
    t_combo_obs: List[List[float]],
    t_dist_obs: List[float],
    reg_term: float,
) -> Tuple[List[float], List[float]]:
    pred_y = []
    pred_y_loss = []
    beta_list = []

    for x_i, t_i in zip(x_est, t_est):
        y_hat_loss, b, _, _, _ = net.forward(x_i, t_i)
        y_hat, _, _, _, _ = net.forward(x_i, t_target)
        pred_y_loss.append(y_hat_loss)
        pred_y.append(y_hat)
        beta_list.append(b)

    pred_y_debiased = []
    for idx, beta_temp in enumerate(beta_list):
        u_target = vector_dot(beta_temp, t_target)
        g_theta = vector_scalar(
            t_target,
            net.c * math.exp(-u_target) / (math.exp(-u_target) + 1) ** 2,
        ) + [1 / (math.exp(-u_target) + 1)]

        u_loss = vector_dot(beta_temp, t_est[idx])
        g_theta_loss = vector_scalar(
            t_est[idx],
            net.c * math.exp(-u_loss) / (math.exp(-u_loss) + 1) ** 2,
        ) + [1 / (math.exp(-u_loss) + 1)]

        lambda_ = [[0.0 for _ in range(len(t_target) + 1)] for _ in range(len(t_target) + 1)]
        for t_obs, dist in zip(t_combo_obs, t_dist_obs):
            u = vector_dot(beta_temp, t_obs)
            g_prime = vector_scalar(
                t_obs,
                net.c * math.exp(-u) / (math.exp(-u) + 1) ** 2,
            ) + [1 / (math.exp(-u) + 1)]
            lambda_ = mat_add(lambda_, mat_scalar(outer(g_prime, g_prime), 2 * dist))

        lambda_reg = mat_add(lambda_, mat_scalar(mat_identity(len(t_target) + 1), reg_term))
        lambda_inv = mat_inv(lambda_reg)
        lambda_inv_loss_prime = mat_vec_mul(lambda_inv, g_theta_loss)
        lambda_inv_loss_prime = vector_scalar(lambda_inv_loss_prime, 2 * (pred_y_loss[idx] - y_est[idx]))
        pred_y_debiased.append(pred_y[idx] - vector_dot(g_theta, lambda_inv_loss_prime))

    return pred_y, pred_y_debiased


def estimate_ate(
    net: SimpleFNN,
    x_est: List[List[float]],
    t_est: List[List[float]],
    y_est: List[float],
    coef_true: List[List[float]],
    c_true: float,
    d_true: float,
    t_target: List[float],
    t_base: List[float],
    t_combo_obs: List[List[float]],
    t_dist_obs: List[float],
    reg_term: float,
) -> Tuple[float, float, float]:
    true_target = generate_y_true_1(coef_true, c_true, d_true, x_est, t_target)
    true_base = generate_y_true_1(coef_true, c_true, d_true, x_est, t_base)
    true_ate = sum(a - b for a, b in zip(true_target, true_base)) / len(x_est)

    pred_target, pred_target_debias = debiased_prediction(
        net,
        x_est,
        t_est,
        y_est,
        t_target,
        t_combo_obs,
        t_dist_obs,
        reg_term,
    )
    pred_base, pred_base_debias = debiased_prediction(
        net,
        x_est,
        t_est,
        y_est,
        t_base,
        t_combo_obs,
        t_dist_obs,
        reg_term,
    )

    sdl_est = sum(a - b for a, b in zip(pred_target, pred_base)) / len(x_est)
    dedl_est = sum(a - b for a, b in zip(pred_target_debias, pred_base_debias)) / len(x_est)

    return true_ate, sdl_est, dedl_est


def fit_linear_regression(x: List[List[float]], t: List[List[float]], y: List[float]) -> List[float]:
    design = [t_i[:] for t_i in t]
    xt = mat_transpose(design)
    xtx = mat_mul(xt, design)
    xty = mat_vec_mul(xt, y)
    coef = mat_vec_mul(mat_inv(xtx), xty)
    return coef


def predict_linear(coef: List[float], x: List[List[float]], t: List[List[float]]) -> List[float]:
    preds = []
    for x_i, t_i in zip(x, t):
        preds.append(vector_dot(coef, t_i))
    return preds


def to_svg_line(points: List[Tuple[float, float]], color: str, dash: str | None = None) -> str:
    if not points:
        return ""
    path = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    dash_attr = f" stroke-dasharray=\"{dash}\"" if dash else ""
    return f"<path d=\"{path}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"2\"{dash_attr} />"


def save_svg(
    filename: str,
    epochs: List[int],
    train_mse: List[float],
    dedl_mape: List[float],
    sdl_mape: List[float],
    lr_mape: List[float],
) -> None:
    width, height = 1000, 450
    margin = 60
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin

    x_min, x_max = min(epochs), max(epochs)
    y1_min, y1_max = min(train_mse), max(train_mse)
    y2_min, y2_max = 0.0, max(dedl_mape + sdl_mape + lr_mape) * 1.1

    def map_x(x):
        return margin + (x - x_min) / (x_max - x_min) * plot_width

    def map_y1(y):
        return margin + plot_height - (y - y1_min) / (y1_max - y1_min) * plot_height

    def map_y2(y):
        return margin + plot_height - (y - y2_min) / (y2_max - y2_min) * plot_height

    lines = []
    lines.append(to_svg_line([(map_x(x), map_y1(y)) for x, y in zip(epochs, train_mse)], "#8c564b"))
    lines.append(to_svg_line([(map_x(x), map_y2(y)) for x, y in zip(epochs, dedl_mape)], "#4c72b0"))
    lines.append(to_svg_line([(map_x(x), map_y2(y)) for x, y in zip(epochs, sdl_mape)], "#55a868", dash="6,4"))
    lines.append(to_svg_line([(map_x(x), map_y2(y)) for x, y in zip(epochs, lr_mape)], "#c44e52", dash="2,4"))

    svg = [
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\">",
        f"<rect width=\"100%\" height=\"100%\" fill=\"white\" />",
        f"<line x1=\"{margin}\" y1=\"{margin}\" x2=\"{margin}\" y2=\"{height - margin}\" stroke=\"#333\" />",
        f"<line x1=\"{width - margin}\" y1=\"{margin}\" x2=\"{width - margin}\" y2=\"{height - margin}\" stroke=\"#333\" />",
        f"<line x1=\"{margin}\" y1=\"{height - margin}\" x2=\"{width - margin}\" y2=\"{height - margin}\" stroke=\"#333\" />",
        f"<text x=\"{width / 2}\" y=\"{height - 15}\" text-anchor=\"middle\" font-size=\"14\">Training Epoch</text>",
        f"<text x=\"20\" y=\"{height / 2}\" text-anchor=\"middle\" font-size=\"14\" transform=\"rotate(-90, 20, {height / 2})\">Training MSE</text>",
        f"<text x=\"{width - 15}\" y=\"{height / 2}\" text-anchor=\"middle\" font-size=\"14\" transform=\"rotate(-90, {width - 15}, {height / 2})\">Estimation MAPE (%)</text>",
    ]
    svg.extend(lines)

    legend_x = width - margin - 160
    legend_y = margin + 10
    legend_items = [
        ("Training MSE", "#8c564b", None),
        ("DeDL MAPE", "#4c72b0", None),
        ("SDL MAPE", "#55a868", "6,4"),
        ("LR MAPE", "#c44e52", "2,4"),
    ]
    for idx, (label, color, dash) in enumerate(legend_items):
        y = legend_y + idx * 18
        dash_attr = f" stroke-dasharray=\"{dash}\"" if dash else ""
        svg.append(f"<line x1=\"{legend_x}\" y1=\"{y}\" x2=\"{legend_x + 20}\" y2=\"{y}\" stroke=\"{color}\" stroke-width=\"2\"{dash_attr} />")
        svg.append(f"<text x=\"{legend_x + 28}\" y=\"{y + 4}\" font-size=\"12\">{label}</text>")

    svg.append("</svg>")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def main() -> None:
    set_seed(7)

    m = 4
    d_c = 8
    n_train = 240
    n_est = 160
    epochs = 400
    lr = 0.05
    reg_term = 0.0005

    _, _, t_combo_obs, t_dist_obs = build_t_combos(m)

    t_base = [0.0 for _ in range(m + 1)]
    t_base[0] = 1.0
    t_target = [0.0 for _ in range(m + 1)]
    t_target[0] = 1.0
    t_target[1] = 1.0
    t_target[2] = 1.0

    coef_true = [[random.uniform(-0.5, 0.5) for _ in range(m + 1)] for _ in range(d_c)]
    c_true = random.uniform(10, 20)
    d_true = 0.0

    x_train = generate_x(d_c, n_train)
    t_train = generate_t(t_combo_obs, t_dist_obs, n_train)
    y_train = generate_y_true(coef_true, c_true, d_true, x_train, t_train)

    x_est = generate_x(d_c, n_est)
    t_est = generate_t(t_combo_obs, t_dist_obs, n_est)
    y_est = generate_y_true(coef_true, c_true, d_true, x_est, t_est)

    coef_lr = fit_linear_regression(x_train, t_train, y_train)
    lr_target = predict_linear(coef_lr, x_est, [t_target for _ in range(n_est)])
    lr_base = predict_linear(coef_lr, x_est, [t_base for _ in range(n_est)])
    true_ate = (
        sum(generate_y_true_1(coef_true, c_true, d_true, x_est, t_target)) / n_est
        - sum(generate_y_true_1(coef_true, c_true, d_true, x_est, t_base)) / n_est
    )
    lr_mape = abs(sum(a - b for a, b in zip(lr_target, lr_base)) / n_est - true_ate) / abs(true_ate)
    print(f"LR MAPE baseline={lr_mape:.3f}")

    net = SimpleFNN(d_c, m, hidden=8)
    net.c = max(y_train)

    epochs_list = []
    train_mse_list = []
    dedl_mape_list = []
    sdl_mape_list = []
    lr_mape_list = []

    for epoch in range(1, epochs + 1):
        train_mse = net.train_epoch(x_train, t_train, y_train, lr)
        true_ate, sdl_est, dedl_est = estimate_ate(
            net,
            x_est,
            t_est,
            y_est,
            coef_true,
            c_true,
            d_true,
            t_target,
            t_base,
            t_combo_obs,
            t_dist_obs,
            reg_term,
        )
        sdl_mape = abs(sdl_est - true_ate) / abs(true_ate)
        dedl_mape = abs(dedl_est - true_ate) / abs(true_ate)

        epochs_list.append(epoch)
        train_mse_list.append(train_mse)
        sdl_mape_list.append(sdl_mape * 100)
        dedl_mape_list.append(dedl_mape * 100)
        lr_mape_list.append(lr_mape * 100)

        if epoch % 50 == 0:
            print(
                f"Epoch {epoch}: train MSE={train_mse:.3f}, "
                f"DeDL MAPE={dedl_mape:.3f}, SDL MAPE={sdl_mape:.3f}"
            )

    os.makedirs("Replication", exist_ok=True)
    save_svg(
        "Replication/figure7_synthetic.svg",
        epochs_list,
        train_mse_list,
        dedl_mape_list,
        sdl_mape_list,
        lr_mape_list,
    )
    print("Saved Replication/figure7_synthetic.svg")


if __name__ == "__main__":
    main()
