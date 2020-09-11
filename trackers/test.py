class Tracker(object):

    def update(self, detections):
        matches, unmatched_tracks, unmatched_detections = self._match(detections)
        return matches, unmatched_tracks, unmatched_detections

    def _match(self, detections):
        print("father match called")
        pass


class Sun(Tracker):
    '''def update(self, detections):
        a, b, c = super(Sun, self).update(detections)
        print(a, b, c)'''

    def _match(self, detections):
        return [1, 2, 3], [4, 5, 6], [7, 8, 9]


def main():
    c = Sun()
    print(c.update([123]))


if __name__ == '__main__':
    main()
