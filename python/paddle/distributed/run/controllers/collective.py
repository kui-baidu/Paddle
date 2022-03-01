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
import six
import time


class CollectiveController(Controller):
    @classmethod
    def enable(cls, ctx):
        if ctx:
            ctx.logger.debug("{} enabled".format(cls.__name__))
            return True
        else:
            return False

    def build_pod(self):
        self.pod.replicas = self.pod_replicas()

        self.pod.rank = self.ctx.args.rank if self.pod.rank < 0 else self.pod.rank

        port = self.ctx.node.get_free_port()

        data = json.dumps({
            'name': self.pod.name,
            'rank': self.pod.rank,
            'replicas': self.pod.replicas,
            'dtype': self.ctx.node.device.dtype,
            'candidate': '{}:{}'.format(self.ctx.node.ip, port)
        })

        peer_list, rank = self.master.sync_peers(
            '/{}/info'.format(self.job.id), self.pod.name, data,
            self.job.replicas, self.pod.rank)

        peer_list = [json.loads(i) for i in peer_list]

        self.ctx.logger.debug("sync peers done {}".format(peer_list))
        self.save_pod_log(peer_list)

        global_size = sum([i['replicas'] for i in peer_list])
        rank_offset = sum([i['replicas'] for i in peer_list[:rank]])

        self.pod.rank = rank
        '''
        The new desinged collective need nothing but a master endpoint
        '''
        collective_master = peer_list[0]['candidate']

        self.pod.reset()
        for i in range(self.pod.replicas):
            e = {
                "PADDLE_MASTER": collective_master,
                "PADDLE_GLOBAL_SIZE": "{}".format(global_size),
                "PADDLE_LOCAL_SIZE": "{}".format(self.pod.replicas),
                "PADDLE_GLOBAL_RANK": "{}".format(i + rank_offset),
                "PADDLE_LOCAL_RANK": "{}".format(i),
            }
            self.add_container(envs=e, log_tag=i)


class CollectiveElasticController(CollectiveController):
    @classmethod
    def enable(cls, ctx):
        if ctx.args.master and ctx.args.master.startswith("etcd://"):
            ctx.logger.debug("{} enabled".format(cls.__name__))
            return True
        else:
            return False

    def register(self):
        if self.job.id == 'default':
            self.ctx.logger.warning(
                'Using default job name may cause conflict, add --id in args')

        self.master.register_heartbeat(self.job.id, self.pod.name)

    def watch(self) -> bool:
        while True:
            # self status
            status = self.pod.watch(timeout=2)
            self.ctx.logger.info("Pod status {}".format(status))
            self.ctx.logger.info("Ctx status {}".format(
                self.ctx.status._current_status))

            if status == self.ctx.status.COMPLETED:
                self.master.set_status(status)
                self.ctx.status.complete()
                self.ctx.logger.info("Pod complete {}".format(status))
                self.pod.stop()
                return True

            # self failure
            elif status == self.ctx.status.FAILED:
                self.master.set_status(status)
                self.master.restart_peer()
                self.ctx.logger.info("Pod failed {}".format(status))
                self.pod.stop()

                if self.ctx.args.elastic_level <= 0:
                    return True
                else:
                    return False

            # peer failure
            if self.ctx.status.need_restart() and self.master.get_status(
            ) != self.ctx.status.COMPLETED:
                self.pod.stop()
                return False

            #peers = self.master.fetch_peer_alive()
            #print("peers {}".format(peers))

    def run(self):

        timeout = self.ctx.args.elastic_timeout if self.job.elastic else self.ctx.args.elastic_timeout * 10
        self.register()

        while self.pod.restart <= self.ctx.args.max_restart:

            self.build_job()

            ok, replicas = self.master.wait_peer_ready(
                self.job.replicas_min, self.job.replicas_max, timeout)
            if ok:
                self.job.replicas = replicas
            else:
                self.ctx.logger.warnning("peer not ready {}".format(self.job))
                break

            self.ctx.logger.debug("Run {}".format(self.job))

            self.build_pod()

            self.master.set_status(self.ctx.status.RUNNING)
            self.ctx.status.run()

            assert len(self.pod.containers) > 0, "No container in the pod"
            self.ctx.logger.debug("Run {}".format(self.pod))
            self.ctx.logger.debug("Run {}".format(self.pod.containers[0]))

            self.pod.deploy()

            if self.watch():
                break

            self.pod.restart += 1

        self.ctx.logger.debug("Job done {}".format(self.job))
