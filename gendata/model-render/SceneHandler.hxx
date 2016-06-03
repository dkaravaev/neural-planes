//
// Created by dmitry on 5/8/16.
//

#ifndef MODEL_RENDER_SCENEHANDLER_HXX
#define MODEL_RENDER_SCENEHANDLER_HXX

#include <osgDB/ReadFile>
#include <osgUtil/Optimizer>
#include <osgGA/TrackballManipulator>
#include <osgGA/KeySwitchMatrixManipulator>
#include <osg/AutoTransform>


#include "SnapImageDrawCallback.hxx"

class SceneHandler
{
public:
    static void to_image(const std::string& model_name, const std::string& filename,
                         const osg::Quat& rotation,
                         uint width, uint height)
    {
        osgViewer::Viewer viewer;
        osg::Node* model = osgDB::readNodeFile(model_name);


        osg::ref_ptr<osgGA::KeySwitchMatrixManipulator> keySwitchManipulator = new osgGA::KeySwitchMatrixManipulator;
        keySwitchManipulator->addMatrixManipulator('1', "Trackball", new osgGA::TrackballManipulator());
        viewer.setCameraManipulator(keySwitchManipulator.get());

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
};

#endif //MODEL_RENDER_SCENEHANDLER_HXX
