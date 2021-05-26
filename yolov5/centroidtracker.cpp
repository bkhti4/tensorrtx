#include "centroidtracker.h"
#include <iterator>

using namespace std;

CentroidTracker::CentroidTracker(int maxDisappeared, int maxDistance, int uniqueClasses) {
    std::vector<int> vect1(uniqueClasses, 0);
    this->nextObjectIDs = vect1;
    this->maxDisappeared = maxDisappeared;
    this->maxDistance = maxDistance;
}

double CentroidTracker::calcDistance(double x1, double y1, double x2, double y2) {
    double x = x1 - x2;
    double y = y1 - y2;
    double dist = sqrt((x * x) + (y * y));       //calculating Euclidean distance

    return dist;
}

void CentroidTracker::register_Object(Object object, int cx, int cy) {
    int & object_ID = this->nextObjectIDs[object.classID];
    UniqueObject uObject;
    uObject.uniqueID = object_ID;
    uObject.cx = cx;
    uObject.cy = cy;
    uObject.trackingcounter = 0;
    uObject.object = object;
    this->instanceObjects.push_back({uObject});
    this->nextObjectIDs[object.classID] += 1;
}

vector<float>::size_type findMin(const vector<float> &v, vector<float>::size_type pos = 0) {
    if (v.size() <= pos) return (v.size());
    vector<float>::size_type min = pos;
    for (vector<float>::size_type i = pos + 1; i < v.size(); i++) {
        if (v[i] < v[min]) min = i;
    }
    return (min);
}

std::vector<UniqueObject> CentroidTracker::update(vector<Object> detObjects) {
    if (detObjects.empty()) {
        auto it = this->instanceObjects.begin();
        while (it != this->instanceObjects.end()) {
            it->trackingcounter++;
            if (it->trackingcounter > this->maxDisappeared) {
                it = this->instanceObjects.erase(it);
            } else {
                ++it;
            }
        }
        return this->instanceObjects;
    }

    // initialize an array of input centroids for the current frame
    vector<pair<int, int>> inputCentroidPairs;
    for (auto objs : detObjects) {
        int cX = int(objs.bboxCords.x + ((objs.bboxCords.width) / 2.0));
        int cY = int(objs.bboxCords.y + ((objs.bboxCords.height) / 2.0));
        inputCentroidPairs.push_back(make_pair(cX, cY));
    }

    //if we are currently not tracking any objects take the input centroids and register each of them
    if (this->instanceObjects.empty()) {
        for (auto objs: detObjects) {
            int tmp_cx = int(objs.bboxCords.x + ((objs.bboxCords.width) / 2.0));
            int tmp_cy = int(objs.bboxCords.y + ((objs.bboxCords.height) / 2.0));
            this->register_Object(objs, tmp_cx, tmp_cy);
        }
    }

        // otherwise, there are currently tracking objects so we need to try to match the
        // input centroids to existing object centroids
    else {
        vector<std::pair<int, int>> objectIDPairs;
        vector<pair<int, int>> objectCentroidPairs;
        for (auto iobject : this->instanceObjects) {
            objectIDPairs.push_back(make_pair(iobject.uniqueID, iobject.object.classID));
            objectCentroidPairs.push_back(make_pair(iobject.cx, iobject.cy));
        }

//        Calculate Distances
        vector<vector<float>> Distances;
        for (int i = 0; i < objectCentroidPairs.size(); ++i) {
            vector<float> temp_D;
            for (vector<vector<int>>::size_type j = 0; j < inputCentroidPairs.size(); ++j) {
                double dist = calcDistance(objectCentroidPairs[i].first, objectCentroidPairs[i].second, inputCentroidPairs[j].first,
                                           inputCentroidPairs[j].second);

                temp_D.push_back(dist);
            }
            Distances.push_back(temp_D);
        }

        // load rows and cols
        vector<int> cols;
        vector<int> rows;

        //find indices for cols
        for (auto v: Distances) {
            auto temp = findMin(v);
            cols.push_back(temp);
        }

        //rows calculation
        //sort each mat row for rows calculation
        vector<vector<float>> D_copy;
        for (auto v: Distances) {
            sort(v.begin(), v.end());
            D_copy.push_back(v);
        }

        // use cols calc to find rows
        // slice first elem of each column
        vector<pair<float, int>> temp_rows;
        int k = 0;
        for (auto i: D_copy) {
            temp_rows.push_back(make_pair(i[0], k));
            k++;
        }
        //print sorted indices of temp_rows
        for (auto const &x : temp_rows) {
            rows.push_back(x.second);
        }

        set<int> usedRows;
        set<int> usedCols;

        //loop over the combination of the (rows, columns) index tuples
        for (int i = 0; i < rows.size(); i++) {
            //if we have already examined either the row or column value before, ignore it
            if (usedRows.count(rows[i]) || usedCols.count(cols[i])) { continue; }

            // Added maxDistance logic here
            if (Distances[rows[i]][cols[i]] > this->maxDistance) { continue; }
            //otherwise, grab the object ID for the current row, set its new centroid,
            // and reset the disappeared counter
            std::pair<int, int> tmpPair = objectIDPairs[rows[i]];
            int objectUniqueID = tmpPair.first;
            int objectClassID = tmpPair.second;
            for (int t = 0; t < this->instanceObjects.size(); t++) {
                if (this->instanceObjects[t].uniqueID == objectUniqueID &&
                        this->instanceObjects[t].object.classID == objectClassID) {
                    this->instanceObjects[t].cx = inputCentroidPairs[cols[i]].first;
                    this->instanceObjects[t].cy = inputCentroidPairs[cols[i]].second;
                    this->instanceObjects[t].trackingcounter = 0;
                    this->instanceObjects[t].object = detObjects[cols[i]];
                }
            }

            usedRows.insert(rows[i]);
            usedCols.insert(cols[i]);
        }

        // compute indexes we have NOT examined yet
        set<int> objRows;
        set<int> inpCols;

        //D.shape[0]
        for (int i = 0; i < objectCentroidPairs.size(); i++) {
            objRows.insert(i);
        }
        //D.shape[1]
        for (int i = 0; i < inputCentroidPairs.size(); i++) {
            inpCols.insert(i);
        }

        set<int> unusedRows;
        set<int> unusedCols;

        set_difference(objRows.begin(), objRows.end(), usedRows.begin(), usedRows.end(),
                       inserter(unusedRows, unusedRows.begin()));
        set_difference(inpCols.begin(), inpCols.end(), usedCols.begin(), usedCols.end(),
                       inserter(unusedCols, unusedCols.begin()));



        //If objCentroids > InpCentroids, we need to check and see if some of these objects have potentially disappeared
        if (objectCentroidPairs.size() >= inputCentroidPairs.size()) {
            // loop over unused row indexes
            for (auto row: unusedRows) {
                std::pair<int, int> tmpPair = objectIDPairs[row];
                int objectUniqueID = tmpPair.first;
                int objectClassID = tmpPair.second;
                auto it = this->instanceObjects.begin();
                while (it != this->instanceObjects.end()) {
                    if (it->uniqueID == objectUniqueID && it->object.classID == objectClassID){
                        it->trackingcounter += 1;
                        if (it->trackingcounter > this->maxDisappeared) {
                            this->instanceObjects.erase(it);
                        }
                        break;
                    }
                    ++it;
                }
            }
        } else {
            for (auto col: unusedCols) {
                this->register_Object(detObjects[col], inputCentroidPairs[col].first, inputCentroidPairs[col].second);
            }
        }

    }

    return this->instanceObjects;
}
