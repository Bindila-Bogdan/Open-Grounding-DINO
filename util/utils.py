import json

import torch


def to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device)
    elif isinstance(item, list):
        return [to_device(i, device) for i in item]
    elif isinstance(item, dict):
        return {k: to_device(v, device) for k, v in item.items()}
    else:
        raise NotImplementedError(
            "Call Shilong if you use other containers! type: {}".format(type(item))
        )


class BestMetricSingle:
    def __init__(self, init_res=0.0, better="large") -> None:
        self.init_res = init_res
        self.best_res = init_res
        self.best_ep = -1

        self.better = better
        assert better in ["large", "small"]

    def isbetter(self, new_res, old_res):
        if self.better == "large":
            return new_res > old_res
        if self.better == "small":
            return new_res < old_res

    def update(self, new_res, ep):
        if self.isbetter(new_res, self.best_res):
            self.best_res = new_res
            self.best_ep = ep
            return True
        return False

    def __str__(self) -> str:
        return "best_res: {}\t best_ep: {}".format(self.best_res, self.best_ep)

    def __repr__(self) -> str:
        return self.__str__()

    def summary(self) -> dict:
        return {
            "best_res": self.best_res,
            "best_ep": self.best_ep,
        }


class BestMetricHolder:
    def __init__(self, init_res=0.0, better="large", use_ema=False) -> None:
        self.best_all = BestMetricSingle(init_res, better)
        self.use_ema = use_ema
        if use_ema:
            self.best_ema = BestMetricSingle(init_res, better)
            self.best_regular = BestMetricSingle(init_res, better)

    def update(self, new_res, epoch, is_ema=False):
        """
        return if the results is the best.
        """
        if not self.use_ema:
            return self.best_all.update(new_res, epoch)
        else:
            if is_ema:
                self.best_ema.update(new_res, epoch)
                return self.best_all.update(new_res, epoch)
            else:
                self.best_regular.update(new_res, epoch)
                return self.best_all.update(new_res, epoch)

    def summary(self):
        if not self.use_ema:
            return self.best_all.summary()

        res = {}
        res.update({f"all_{k}": v for k, v in self.best_all.summary().items()})
        res.update({f"regular_{k}": v for k, v in self.best_regular.summary().items()})
        res.update({f"ema_{k}": v for k, v in self.best_ema.summary().items()})
        return res

    def __repr__(self) -> str:
        return json.dumps(self.summary(), indent=2)

    def __str__(self) -> str:
        return self.__repr__()
