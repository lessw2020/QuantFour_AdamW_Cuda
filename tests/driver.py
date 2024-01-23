# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import pytest
import torch
import torch.nn as nn
import torch.optim as torch_optim
import sys
sys.path.append("..")

from utils_test import assert_expected, gpu_test, set_rng_seed

from optim.fused_quantfour import AdamWFused_QuantFour

@pytest.fixture(autouse=True)
def random():
    set_rng_seed(2020)


class TestAdamw4Bit_Optimizer:
    def _test_basics(self, model, model_clone):
        # Test non-default options
        betas = (0.8, 0.88)
        weight_decay = 0.03
        lr = 0.005
        eps = 1e-8

        adam_opt = torch_optim.AdamW(
            model_clone.parameters(),lr=lr, betas=betas, weight_decay=weight_decay, eps=eps,
        )
        fourbit_adamw_opt = AdamWFused_QuantFour(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,)

        # Verify params are equal initially
        model_orig_params = [p.clone() for p in model.parameters()]
        #print(f"{model_orig_params=}")
        for p1, p2 in zip(model_clone.parameters(), model_orig_params):
            assert_expected(p1, p2)
        print(f"len model {len(model_orig_params)}")
        _size = 5

        for i in range(5):
            adam_opt.zero_grad(set_to_none=True)
            fourbit_adamw_opt.zero_grad(set_to_none=True)
            inp = torch.randn(_size, _size, device=next(model.parameters()).device)
            model(inp).sum().backward()
            model_clone(inp).sum().backward()
            adam_opt.step()
            fourbit_adamw_opt.step()

        # Ensure params are modified from original
            if i==0:
                for p1, p2 in zip(model.parameters(), model_orig_params):
                    assert not torch.equal(p1, p2)
                    #print(f"{p1[0:10]=},")
                    #print(f"{p2[0:10]=}")
                print(f"confirm modified params")
        # confirm we match AdamW at each step
            for p1, p2 in zip(model.parameters(), model_clone.parameters()):
                assert_expected(p1, p2)
            print(f"quantfour matches adamw in step {i}")



    def _test_adam_equivalence(self, model, model_clone, config_path):
        # Test non-default options
        betas = (0.8, 0.88)
        weight_decay = 0.03

        adam_opt = torch_optim.AdamW(
            model_clone.parameters(), betas=betas, weight_decay=weight_decay
        )
        fourbit_adamw_opt = AdamW_FourBit(
            model.parameters(),
            betas=betas,
            weight_decay=weight_decay,
            qconfig=config_path,
        )


        # Verify params are equal initially
        model_orig_params = [p.clone() for p in model.parameters()]
        for p1, p2 in zip(model_clone.parameters(), model_orig_params):
            assert_expected(p1, p2)

        for i in range(6):
            if i % 2:
                adam_opt.zero_grad(set_to_none=True)
                fourbit_adamw_opt.zero_grad(set_to_none=True)
            else:
                adam_opt.zero_grad(set_to_none=False)
                fourbit_adamw_opt.zero_grad(set_to_none=False)

            inp = torch.randn(4096, 4096, device=next(model.parameters()).device)
            model(inp).sum().backward()
            model_clone(inp).sum().backward()
            adam_opt.step()
            fourbit_adamw_opt.step()

            # Ensure params are modified from original
            if i == 0:
                for p1, p2 in zip(model.parameters(), model_orig_params):
                    assert not torch.equal(p1, p2)

            for p1, p2 in zip(model.parameters(), model_clone.parameters()):
                assert_expected(p1, p2)

    @gpu_test()
    def test_adam_equivalence_gpu(self, device="cuda"):
        """
        Tests, on gpu, that fourbit_adamw_opt is approx equivalent to AdamW
        """

        # model = nn.Sequential(nn.Linear(5, 10), nn.Linear(10, 10), nn.Linear(10, 5))
        model = nn.Sequential(nn.Linear(5, 10), nn.Linear(10, 5))
        model.cuda()

        model_clone = deepcopy(model)

        #self._test_adam_equivalence(model, model_clone, config_path)
        self._test_basics(model, model_clone)


    '''def test_adam_equivalence_cpu(self, config_path: None, device="cpu", ):
        """
        Tests that fourbit is equivalent to AdamW on cpu
        """
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip(reason="CUDA not available")

        model = nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 5), nn.Linear(5, 5))
        if device == "cuda":
            model.cuda()

        model_clone = deepcopy(model)

        self._test_adam_equivalence(model, model_clone, config_path)
    '''
