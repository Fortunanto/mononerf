class OptimizerGroup:
    def __init__(self):
        self.optimizers = []
        self.schedulers = []

    def add(self, optimizer, scheduler=None):
        self.optimizers.append(optimizer)
        if scheduler:
            self.schedulers.append(scheduler)
        else:
            self.schedulers.append(None)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def scheduler_step(self):
        for scheduler in self.schedulers:
            if scheduler is not None:
                scheduler.step()

    def get_lr(self):
        lrs = []
        for optimizer in self.optimizers:
            lrs.append(optimizer.param_groups[0]['lr'])
        return lrs
