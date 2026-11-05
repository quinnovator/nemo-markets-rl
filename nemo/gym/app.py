"""NeMo Gym resources server for the deadline market.

Written against the Gym version bundled in NVIDIA's NeMo RL 0.7.0 container.
`prepare.py` copies this file into Gym as `resources_servers/deadline_offer/app.py`,
next to an unchanged copy of `market.py`, which defines the reward.
"""

import json

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from pydantic import ConfigDict

import market


class DeadlineOfferConfig(BaseResourcesServerConfig):
    pass


class VerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    task: dict  # market.Task as JSON, carried in each dataset row


class VerifyResponse(BaseVerifyResponse):
    task: dict
    valid_action: bool
    offer: float | None
    optimal_offer: float | None
    expected_surplus: float
    optimal_surplus: float
    regret: float


class DeadlineOfferServer(SimpleResourcesServer):
    config: DeadlineOfferConfig

    async def verify(self, body: VerifyRequest) -> VerifyResponse:
        task = market.task_from_json(body.task)
        text = "".join(
            content.text
            for item in body.response.output
            if item.type == "message"
            for content in item.content
            if content.type == "output_text"
        )
        valid, offer = market.parse_action(text)
        if not valid:
            print("INVALID_ACTION " + json.dumps({"text": text}), flush=True)
        best_offer, best_ev = market.optimal_offer(task)
        ev = market.expected_surplus(task, offer) if valid else 0.0
        return VerifyResponse(
            **body.model_dump(),
            reward=market.reward(task, text),
            valid_action=valid,
            offer=offer,
            optimal_offer=best_offer,
            expected_surplus=ev,
            optimal_surplus=best_ev,
            regret=(best_ev - ev) / task.public_value,
        )


if __name__ == "__main__":
    DeadlineOfferServer.run_webserver()
