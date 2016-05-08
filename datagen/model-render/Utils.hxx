//
// Created by dmitry on 5/8/16.
//

#ifndef MODEL_RENDER_UTILS_HXX
#define MODEL_RENDER_UTILS_HXX

#include <utility>
#include <string>
#include <sstream>
#include <cmath>
#include <ctime>
#include <cstdlib>

#include <osg/Quat>

class Utils
{
public:
    static osg::Quat create_quat(int x, int y, int z)
    {

        auto q_bank = osg::Quat(std::sin(x / 2), 0, 0, std::cos(x / 2));
        auto q_alt = osg::Quat(0, std::sin(y / 2), 0, std::cos(z / 2));
        auto q_heading = osg::Quat(0, 0, std::sin(z / 2), std::cos(z / 2));

        return (q_heading * q_alt) * q_bank;
    }
};
#endif //MODEL_RENDER_UTILS_HXX
