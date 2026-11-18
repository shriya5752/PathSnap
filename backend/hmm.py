"""Viterbi decoding on candidate road segments with explicit trip breaks."""

from dataclasses import dataclass
from math import log

import numpy as np

from .routing import DecoderConfig, RoadGraph


@dataclass
class Decoding:
    point_arcs: list[int | None]
    route_blocks: list[list[int]]
    valid_links: int


@dataclass
class Comparison:
    decoded: Decoding
    classifier: Decoding
    eligible_links: int


def _append_link(blocks, current, link):
    if link is None:
        blocks.append([current])
    else:
        blocks[-1].extend(link.extra_arcs)


def decode_trip(
    candidates, graph: RoadGraph, points, times, config: DecoderConfig = DecoderConfig()
):
    size = len(candidates)
    decoded = Decoding([None] * size, [], 0)
    classifier = Decoding([None] * size, [], 0)
    eligible_links = 0
    scores = {}
    steps = []

    def finish_block():
        if not steps:
            return
        best = max(scores, key=scores.__getitem__)
        block = []
        for index, back in reversed(steps):
            decoded.point_arcs[index] = best
            if back:
                previous, link = back[best]
                block.extend(reversed(link.extra_arcs))
                best = previous
                decoded.valid_links += 1
        block.append(best)
        decoded.route_blocks.append(list(reversed(block)))
        steps.clear()

    for index, group in enumerate(candidates):
        if not group:
            finish_block()
            scores = {}
            continue

        emissions = {item.arc_id: log(max(item.score, 1e-12)) for item in group}
        winner = max(emissions, key=emissions.__getitem__)
        classifier.point_arcs[index] = winner
        links = {}
        if index and candidates[index - 1]:
            gap = float(times[index] - times[index - 1])
            if 0 < gap <= config.max_gap_s:
                eligible_links += 1
                displacement = float(np.linalg.norm(points[index] - points[index - 1]))
                links = graph.links(candidates[index - 1], group, gap, displacement, config)
        classifier_link = links.get((classifier.point_arcs[index - 1], winner)) if index else None
        _append_link(classifier.route_blocks, winner, classifier_link)
        classifier.valid_links += int(classifier_link is not None)

        best_scores = {}
        back = {}
        for (before, after), link in links.items():
            if before not in scores:
                continue
            score = scores[before] + link.log_score + emissions[after]
            if after not in best_scores or score > best_scores[after]:
                best_scores[after] = score
                back[after] = before, link
        if not back:
            finish_block()
            scores = emissions
        else:
            scores = best_scores
        steps.append((index, back))

    finish_block()
    return Comparison(decoded, classifier, eligible_links)
