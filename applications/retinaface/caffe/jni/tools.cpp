#include "tools.h"

void nms_cpu(std::vector<Anchor>& boxes, float threshold, std::vector<Anchor>& filterOutBoxes) {
    filterOutBoxes.clear();
    if(boxes.size() == 0)
        return;
    std::vector<size_t> idx(boxes.size());

    for(unsigned i = 0; i < idx.size(); i++)
    {
        idx[i] = i;
    }

    //descending sort
    sort(boxes.begin(), boxes.end(), std::greater<Anchor>());

    while(idx.size() > 0)
    {
        int good_idx = idx[0];
        filterOutBoxes.push_back(boxes[good_idx]);

        std::vector<size_t> tmp = idx;
        idx.clear();
        for(unsigned i = 1; i < tmp.size(); i++)
        {
            int tmp_i = tmp[i];
            float inter_x1 = std::max( boxes[good_idx][0], boxes[tmp_i][0] );
            float inter_y1 = std::max( boxes[good_idx][1], boxes[tmp_i][1] );
            float inter_x2 = std::min( boxes[good_idx][2], boxes[tmp_i][2] );
            float inter_y2 = std::min( boxes[good_idx][3], boxes[tmp_i][3] );

            float w = std::max((inter_x2 - inter_x1 + 1), 0.0F);
            float h = std::max((inter_y2 - inter_y1 + 1), 0.0F);

            float inter_area = w * h;
            float area_1 = (boxes[good_idx][2] - boxes[good_idx][0] + 1) * (boxes[good_idx][3] - boxes[good_idx][1] + 1);
            float area_2 = (boxes[tmp_i][2] - boxes[tmp_i][0] + 1) * (boxes[tmp_i][3] - boxes[tmp_i][1] + 1);
            float o = inter_area / (area_1 + area_2 - inter_area);           
            if( o <= threshold )
                idx.push_back(tmp_i);
        }
    }
}