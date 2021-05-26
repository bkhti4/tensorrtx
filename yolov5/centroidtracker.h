
#ifndef C___CENTROIDTRACKER_H
#define C___CENTROIDTRACKER_H

#endif //C___CENTROIDTRACKER_H

#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <set>
#include <algorithm>
#include <opencv2/opencv.hpp>


class Object{
    public:
        int classID;
        cv::Rect bboxCords;
        double detConfidence;
};

class UniqueObject{
    public:
        int uniqueID;
        int cx;
        int cy;
        int trackingcounter;
        Object object;
};


class CentroidTracker {
public:
    explicit CentroidTracker(int maxDisappeared, int maxDistance, int uniqueClasses);

    void register_Object(Object object, int cx, int cy);

    std::vector<UniqueObject> update(std::vector<Object> detObjects);

    // <ID, centroids>
    std::vector<UniqueObject> instanceObjects;

private:
    int maxDisappeared;
    int maxDistance;

    std::vector<int> nextObjectIDs;

    static double calcDistance(double x1, double y1, double x2, double y2);
};
