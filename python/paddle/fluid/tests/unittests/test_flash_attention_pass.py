# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import unittest

import numpy as np

import paddle
import paddle.fluid.core as core
import paddle.nn.functional as F
from paddle.distributed.passes import PassManager, new_pass


def get_cuda_version():
    result = os.popen("nvcc --version").read()
    regex = r'release (\S+),'
    match = re.search(regex, result)
    if match:
        num = str(match.group(1))
        integer, decimal = num.split('.')
        return int(integer) * 1000 + int(float(decimal) * 10)
    else:
        return -1


def nn(qkv, scale_pos=1, causal=False, dropout_prob=0.0):
    qkv = paddle.transpose(qkv, [0, 2, 1, 3])
    qt, kt, vt = paddle.split(qkv, num_or_sections=3, axis=-1)

    if scale_pos == 1:
        scale = 1.0 / np.sqrt(qt.shape[-1])
        qt = paddle.scale(qt, scale)
    elif scale_pos == 2:
        scale = 1.0 / np.sqrt(kt.shape[-1])
        kt = paddle.scale(kt, scale)

    s = paddle.matmul(qt, kt, transpose_y=True)

    if scale_pos == 3:
        scale = 1.0 / np.sqrt(kt.shape[-1])
        s = paddle.scale(s, scale)

    p = (
        paddle.incubate.softmax_mask_fuse_upper_triangle(s)
        if causal
        else F.softmax(s)
    )
    if dropout_prob > 0:
        p = paddle.nn.functional.dropout(p, dropout_prob)
    o = paddle.matmul(p, vt)
    o = paddle.transpose(o, [0, 2, 1, 3])
    o = paddle.reshape(o, [0, 0, o.shape[2] * o.shape[3]])
    return o


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11030,
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.3",
)
class TestFlashAttentionPass(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.num_head = 2
        self.head_dim = 8
        self.bs = 2
        self.seq_len = 8

        self.shape = (self.bs, self.seq_len, self.num_head * self.head_dim * 3)
        self.dtype = 'float16'
        self.causal = True
        self.dropout = 0.1

    def test_pass(self):
        data = np.random.random(self.shape)

        out_ref = self.run_program(data, apply_pass=False)
        print("out_ref", out_ref)
        out_pass = self.run_program(data, apply_pass=True)
        print("out_pass", out_pass)
        '''
        np.testing.assert_allclose(
            fetches_result[0], out_, rtol=5e-03, atol=1e-03
        )
        '''

    def run_program(self, data, apply_pass=False):

        paddle.enable_static()
        paddle.seed(1024)

        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()

        with paddle.static.program_guard(main_prog, startup_prog):
            qkv = paddle.static.data(
                name="qkv", shape=self.shape, dtype=self.dtype
            )

            qkv = paddle.reshape(qkv, [0, 0, self.num_head, 3 * self.head_dim])
            out = nn(
                qkv, scale_pos=3, causal=self.causal, dropout_prob=self.dropout
            )
            out = paddle.reshape(out, [-1])
            loss = paddle.mean(out)

            # sgd_optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.001)
            # sgd_optimizer.minimize(loss)

            if apply_pass:
                pass_manager = PassManager([new_pass("flash_attention")])
                pass_manager.apply([main_prog], [startup_prog])

                ops = main_prog.global_block().ops
                assert "flash_attn" in [op.type for op in ops]

            exe = paddle.static.Executor(self.place)
            exe.run(startup_prog)

            ret_loss = exe.run(
                main_prog,
                feed={
                    "qkv": data.astype('float16'),
                },
                fetch_list=[out.name],
            )

            # print("prog", main_prog)
            return ret_loss


'''
class TestFlashAttentionPass1(TestFlashAttentionPass):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.num_head = 8
        self.head_dim = 8
        self.bs = 2
        self.seq_len = 64

        self.shape = (self.bs, self.seq_len, self.num_head * self.head_dim * 3)
        self.dtype = 'float16'
        self.causal = True
        self.dropout = 0.1
'''


if __name__ == '__main__':
    unittest.main()
