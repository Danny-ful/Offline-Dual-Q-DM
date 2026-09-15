from types import SimpleNamespace
import unittest

import torch

from agent.lr_scheduler import make_actor_lr_scheduler


def _args(enabled=True, final_lr=3e-6, anneal_updates=3):
    return SimpleNamespace(
        schedular=False,
        actor_lr_scheduler=SimpleNamespace(
            enabled=enabled,
            final_lr=final_lr,
            anneal_updates=anneal_updates,
        ),
    )


class ActorLRSchedulerTest(unittest.TestCase):
    def test_linear_schedule_reaches_absolute_final_lr(self):
        for initial_lr in (1e-4, 3e-5, 1e-5):
            with self.subTest(initial_lr=initial_lr):
                parameter = torch.nn.Parameter(torch.zeros(()))
                optimizer = torch.optim.Adam([parameter], lr=initial_lr)
                scheduler = make_actor_lr_scheduler(
                    optimizer, _args(), initial_lr
                )

                observed = [optimizer.param_groups[0]["lr"]]
                for _ in range(4):
                    optimizer.step()
                    scheduler.step()
                    observed.append(optimizer.param_groups[0]["lr"])

                self.assertAlmostEqual(observed[0], initial_lr)
                self.assertAlmostEqual(observed[3], 3e-6)
                self.assertAlmostEqual(observed[4], 3e-6)
                self.assertEqual(observed, sorted(observed, reverse=True))

    def test_scheduler_can_reach_zero(self):
        parameter = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.Adam([parameter], lr=1e-4)
        scheduler = make_actor_lr_scheduler(
            optimizer,
            _args(final_lr=0.0, anneal_updates=2),
            initial_lr=1e-4,
        )

        for _ in range(2):
            optimizer.step()
            scheduler.step()

        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)

    def test_disabled_scheduler_is_none(self):
        parameter = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.Adam([parameter], lr=1e-4)

        self.assertIsNone(make_actor_lr_scheduler(
            optimizer,
            _args(enabled=False),
            initial_lr=1e-4,
        ))


if __name__ == "__main__":
    unittest.main()
