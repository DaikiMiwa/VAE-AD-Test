import numpy as np
import tensorflow as tf
from mpmath import mp
from si4dnn import si
from si4dnn.layers import truncated_interval
from sicore import NaiveInferenceNorm, SelectiveInferenceNorm

from si4vae import util


class AEInferenceNorm(si.SI4DNN):
    def __init__(
        self, model, thr: float, var: float | np.ndarray = 1, smoothing=None, kernel_size=3
    ):
        super().__init__(model, var)
        self.thr = thr
        self.smoothing = smoothing
        if self.smoothing not in [None, "MEAN", "GAUSSIAN"]:
            raise AttributeError("smoothing must be None, 'MEAN', 'GAUSSIAN'")

        if self.smoothing == "MEAN":
            self.kernel = util.make_mean_filter_kernel(kernel_size)
        elif self.smoothing == "GAUSSIAN":
            self.kernel = util.make_gaussian_filter_kernel(kernel_size, 1)

    def construct_hypothesis(self, X):
        self.shape = X.shape
        self.output, _ = self.si_model.forward(X)
        self.input = X

        input_vec = tf.cast(tf.reshape(X, [-1]), tf.float64)
        output_vec = tf.cast(tf.reshape(self.output, [-1]), tf.float64)

        abnormal_error = self.input - self.output

        if self.smoothing is not None:
            smoothed_abnormal_error = tf.nn.depthwise_conv2d(
                abnormal_error, self.kernel, [1, 1, 1, 1], "SAME"
            )
        else:
            smoothed_abnormal_error = abnormal_error

        # とりあえずモデルを取得
        abnormal_index = tf.where(
            tf.abs(smoothed_abnormal_error) >= self.thr, True, False
        )
        self.abnormal_index = tf.reshape(abnormal_index, [-1])

        number_of_abnormal_index = tf.reduce_sum(
            tf.cast(self.abnormal_index, dtype=tf.int32)
        )

        # Check wether hypothesis is obtained or not
        if 0 == number_of_abnormal_index or number_of_abnormal_index == tf.size(
            self.abnormal_index
        ):
            raise si.NoHypothesisError

        number_abnormal_pixel = tf.reduce_sum(tf.cast(abnormal_index, tf.float64))
        number_normal_pixel = tf.reduce_sum(
            tf.cast(abnormal_index == False, tf.float64)
        )

        eta_normal = (
            tf.cast(tf.where(self.abnormal_index, 0.0, 1.0), tf.float64)
            / number_normal_pixel
        )
        eta_abnormal = (
            tf.cast(tf.where(self.abnormal_index, 1.0, 0.0), tf.float64)
            / number_abnormal_pixel
        )
        self.eta = eta_normal - eta_abnormal

        # 保存
        self.input_vec = input_vec
        self.output_vec = output_vec

        self.si_calculator = SelectiveInferenceNorm(
            self.input_vec, self.var, self.eta, use_tf=True
        )

        self.si_calculator_naive = NaiveInferenceNorm(
            self.input_vec, self.var, self.eta, use_tf=True
        )

        # set upper and lower bound of search range
        sd: float = np.sqrt(self.si_calculator.eta_sigma_eta)
        self.max_tail = sd * 100

    def model_selector(self, abnormal_index):
        return tf.reduce_all(
            tf.math.equal(self.abnormal_index, tf.reshape(abnormal_index, [-1]))
        )

    def algorithm(self, a, b, z):
        x = a + b * z
        B, H, W, C = self.shape

        input_x = tf.reshape(tf.constant(x, dtype=tf.float64), [B, H, W, C])
        input_bias = tf.zeros([B, H, W, C], dtype=tf.float64)
        input_a = tf.reshape(tf.constant(a, dtype=tf.float64), [B, H, W, C])
        input_b = tf.reshape(tf.constant(b, dtype=tf.float64), [B, H, W, C])
        input_si = (
            input_x,
            input_bias,
            input_a,
            input_b,
            -1 * self.max_tail,
            self.max_tail,
        )

        l, u, output, output_si_dict = self.si_model.forward_si(input_si)

        _output_x, _ = self.si_model.forward(input_x)

        # check two vector is close
        # assert np.allclose(_output_x, output[0])

        # 最後の出力を取り出す
        output_x, output_bias, output_a, output_b, l, u = output_si_dict[
            self.si_model.output
        ]
        output_x = output_x[0]
        output_bias = output_bias[0]
        output_a = output_a[0]
        output_b = output_b[0]
        l = l[0]
        u = u[0]

        if l > u:
            assert False

        # selection event of reconstruction error
        error_x = input_x - output_x
        error_bias = 0 - output_bias
        error_a = input_a - output_a
        error_b = input_b - output_b

        # apply smoothing
        if self.smoothing is not None:
            smoothed_error_x = tf.nn.depthwise_conv2d(
                error_x, self.kernel, [1, 1, 1, 1], "SAME"
            )
            smoothed_error_bias = tf.nn.depthwise_conv2d(
                error_bias, self.kernel, [1, 1, 1, 1], "SAME"
            )
            smoothed_error_a = tf.nn.depthwise_conv2d(
                error_a, self.kernel, [1, 1, 1, 1], "SAME"
            )
            smoothed_error_b = tf.nn.depthwise_conv2d(
                error_b, self.kernel, [1, 1, 1, 1], "SAME"
            )
            abnormal_index = tf.abs(smoothed_error_x) >= self.thr
        else:
            smoothed_error_x = error_x
            smoothed_error_bias = error_bias
            smoothed_error_a = error_a
            smoothed_error_b = error_b
            abnormal_index = tf.abs(error_x) >= self.thr

        # positive reconstruction error
        positive_index = smoothed_error_x >= self.thr

        tTa = tf.where(positive_index, -smoothed_error_a, smoothed_error_a)
        tTb = tTb = tf.where(positive_index, -smoothed_error_b, smoothed_error_b)
        event_bias = smoothed_error_bias - self.thr
        event_bias = tf.where(positive_index, -event_bias, event_bias)
        l_positive, u_positive = truncated_interval(tTa, tTb, event_bias)

        assert l_positive < u_positive

        # negative reconstruction error
        negative_index = smoothed_error_x < -self.thr
        tTa = tf.where(negative_index, smoothed_error_a, -smoothed_error_a)
        tTb = tf.where(negative_index, smoothed_error_b, -smoothed_error_b)
        event_bias = smoothed_error_bias + self.thr
        event_bias = tf.where(negative_index, event_bias, -event_bias)

        l_negative, u_negative = truncated_interval(tTa, tTb, event_bias)
        assert l_negative < u_negative

        l = tf.reduce_max([l_positive, l_negative, l])
        u = tf.reduce_min([u_positive, u_negative, u])

        if l > u:
            print("negative", l_negative, u_negative)
            print("positive", l_positive, u_positive)
            print("normal", l, u)
            assert l < u

        return abnormal_index, (l, u)

    def naive_p_value(self):
        stat = self.si_calculator.stat.numpy()
        std = np.sqrt(self.si_calculator.eta_sigma_eta.numpy())

        # compute let cdf and right cdf using mpmath
        mp.dps = 3000
        left_prob = mp.ncdf(-np.abs(stat) / std)
        right_prob = 1 - mp.ncdf(np.abs(stat) / std)

        p_value = left_prob + right_prob

        return p_value

    def naive_inference(self, X):
        self.construct_hypothesis(X)
        p_value = self.naive_p_value()

        return float(p_value)

    def bonf_inference(self, X):
        self.construct_hypothesis(X)
        self.bonf_coef = mp.mpf(2) ** (self.abnormal_index.shape[0])
        p_value = min(float(self.naive_p_value() * self.bonf_coef),1)

        return p_value
