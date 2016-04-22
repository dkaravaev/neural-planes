#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgUtil/Optimizer>
#include <osgViewer/Viewer>
#include <osgGA/TrackballManipulator>
#include <osgGA/KeySwitchMatrixManipulator>
#include <osg/AutoTransform>


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

void scene_to_image(osg::Node* model, const std::string& filename)
{
    osgViewer::Viewer viewer;

    osg::ref_ptr<osgGA::KeySwitchMatrixManipulator> keyswitchManipulator = new osgGA::KeySwitchMatrixManipulator;
    keyswitchManipulator->addMatrixManipulator('1', "Trackball", new osgGA::TrackballManipulator());
    viewer.setCameraManipulator(keyswitchManipulator.get());

    uint width = 200;
    uint height = 200;

    osg::ref_ptr<osg::GraphicsContext> pbuffer;

    osg::ref_ptr<osg::GraphicsContext::Traits> traits = new osg::GraphicsContext::Traits;
    {
        traits->x = 0;
        traits->y = 0;
        traits->width = width;
        traits->height = height;
        traits->red = 8;
        traits->green = 8;
        traits->blue = 8;
        traits->alpha = 8;
        traits->windowDecoration = false;
        traits->pbuffer = true;
        traits->doubleBuffer = true;
        traits->sharedContext = 0;
    }

    pbuffer = osg::GraphicsContext::createGraphicsContext(traits.get());

    osgUtil::Optimizer optimizer;
    optimizer.optimize(model);

    osg::AutoTransform* trans = new osg::AutoTransform;
    trans->addChild(model);
    trans->setRotation(osg::Quat(0.6, -1.5, 0.5, -0.7));

    viewer.setSceneData(trans);
    viewer.getCamera()->setClearColor(osg::Vec4(1,1,1,1));

    osg::ref_ptr<osg::Camera> camera = new osg::Camera;
    camera->setGraphicsContext(pbuffer.get());
    camera->setViewport(new osg::Viewport(0,0,width,height));

    GLenum buffer = pbuffer->getTraits()->doubleBuffer ? GL_BACK : GL_FRONT;
    camera->setDrawBuffer(buffer);
    camera->setReadBuffer(buffer);
    camera->setFinalDrawCallback(new SnapImageDrawCallback(filename));

    viewer.addSlave(camera.get(), osg::Matrixd(), osg::Matrixd());

    viewer.realize();

    viewer.frame();
}

int main(int argc, char** argv)
{
    osg::Node* model = osgDB::readNodeFile("/home/dmitry/Downloads/F-14A_Tomcat/F-14A_Tomcat.obj");
    scene_to_image(model, "/home/dmitry/plane.png");

    return 0;
}




