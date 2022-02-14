# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from .controller import Controller

import json
import os


class CollectiveController(Controller):
    @classmethod
    def enable(cls, ctx):
        if ctx:
            ctx.logger.debug("CollectiveController enabled")
            return True
        else:
            return False

    def build_job(self):
        super().build_job()

    def build_pod(self):
        self.pod.replicas = self.pod_replicas()
        self.pod.rank = self.ctx.args.rank

        port = self.ctx.node.get_free_port()
        ip = self.ctx.node.ip

        data = json.dumps({
            'name': self.pod.name,
            'rank': self.pod.rank,
            'replicas': self.pod.replicas,
            'dtype': self.ctx.node.device.dtype,
            'candidate': '{}:{}'.format(ip, port)
        })

        peer_list, rank = self.master.sync_peers(
            '/{}/info'.format(self.job.id), self.pod.name, data,
            self.job.replicas, self.pod.rank)

        peer_list = [json.loads(i) for i in peer_list]

        self.save_log(peer_list)

        global_size = sum([i['replicas'] for i in peer_list])
        rank_offset = sum([i['replicas'] for i in peer_list[:rank]])

        self.pod.rank = rank
        '''
        The new desinged collective need nothing but a master endpoint
        '''
        collective_master = peer_list[0]['candidate']

        for i in range(self.pod.replicas):
            e = {
                "PADDLE_MASTER": collective_master,
                "PADDLE_GLOBAL_SIZE": "{}".format(global_size),
                "PADDLE_LOCAL_SIZE": "{}".format(self.pod.replicas),
                "PADDLE_GLOBAL_RANK": "{}".format(i + rank_offset),
                "PADDLE_LOCAL_RANK": "{}".format(i),
            }
            log_file = "worker.{}.{}.log".format(self.pod.name, i)
            self.add_container(envs=e, log_file=log_file)

    '''
    compatible version of build_pod
    '''

    def _build_pod(self):

        self.pod.replicas = self.pod_replicas()

        self.pod.endpoints = [
            "{}:{}".format(self.ctx.node.ip, p)
            for p in self.ctx.node.get_free_ports(self.pod.replicas)
        ]

        eps, _ = self.master.sync_peers(
            '/workers',
            self.pod.name,
            ",".join(self.pod.endpoints),
            self.job.replicas, )

        self.job.endpoints = ",".join(eps).split(",")

        for i in range(self.pod.replicas):
            e = {
                "PADDLE_TRAINER_ENDPOINTS": ",".join(self.job.endpoints),
                "PADDLE_CURRENT_ENDPOINT": self.pod.endpoints[i],
                "PADDLE_TRAINER_ID":
                "%d" % self.job.endpoints.index(self.pod.endpoints[i]),
                "PADDLE_TRAINERS_NUM": "%d" % len(self.job.endpoints),
                "PADDLE_RANK_IN_NODE": str(i),
            }
            c = self.build_container(envs=e)
            self.add_container(c)
