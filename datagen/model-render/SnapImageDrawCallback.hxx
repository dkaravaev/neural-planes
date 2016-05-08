//
// Created by dmitry on 5/8/16.
//

#ifndef MODEL_RENDER_SNAPIMAGEDRAWCALLBACK_HXX
#define MODEL_RENDER_SNAPIMAGEDRAWCALLBACK_HXX

#include <string>

#include <osgViewer/Viewer>
#include <osgDB/WriteFile>

class SnapImageDrawCallback : public osg::Camera::DrawCallback
{
public:

    SnapImageDrawCallback(const std::string& filename)
            : filename(filename)
    {

    }

    virtual void operator ()(const osg::Camera& camera) const
    {
        int x, y;
        int width,height;
        x = static_cast<int>(camera.getViewport()->x());
        y = static_cast<int>(camera.getViewport()->y());
        width = static_cast<int>(camera.getViewport()->width());
        height = static_cast<int>(camera.getViewport()->height());

        osg::ref_ptr<osg::Image> image = new osg::Image;
        image->readPixels(x,y,width,height,GL_RGB,GL_UNSIGNED_BYTE);

        osgDB::writeImageFile(*image, filename);
    }

protected:
    std::string filename;
};

#endif //MODEL_RENDER_SNAPIMAGEDRAWCALLBACK_HXX
