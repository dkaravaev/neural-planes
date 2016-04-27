#include <sstream>
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

void scene_to_image(osg::Node* model, const std::string& filename, const osg::Quat& rotation)
{
    osgViewer::Viewer viewer;

    osg::ref_ptr<osgGA::KeySwitchMatrixManipulator> keyswitchManipulator = new osgGA::KeySwitchMatrixManipulator;
    keyswitchManipulator->addMatrixManipulator('1', "Trackball", new osgGA::TrackballManipulator());
    viewer.setCameraManipulator(keyswitchManipulator.get());

    uint width = 120;
    uint height = 120;

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
    trans->setRotation(rotation);

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


osg::Quat parse_quat(const std::string& str)
{
    std::istringstream iss(str);
    std::string sub;
    iss >> sub;
    double a = std::stod(sub);
    iss >> sub;
    double b = std::stod(sub);
    iss >> sub;
    double c = std::stod(sub);
    iss >> sub;
    double d = std::stod(sub);

    // osg::Quat(0.6, -1.5, 0.5, -0.7)
    return osg::Quat(a, b, c, d);
}

// argv[1] = model name
// argv[2] = result name
// argv[3] = rotation string
int main(int argc, char** argv)
{
    // osg::Node* model = osgDB::readNodeFile("/home/dmitry/Downloads/F-14A_Tomcat/F-14A_Tomcat.obj");
    // scene_to_image(model, "/home/dmitry/plane.png");

    osg::Node* model = osgDB::readNodeFile(argv[1]);
    scene_to_image(model, argv[2], parse_quat(argv[3]));

    return 0;
}




