package com.findwise.gateembeddedexample;

import gate.Corpus;
import gate.Document;
import gate.Factory;
import gate.Gate;
import gate.creole.ExecutionException;
import gate.creole.ResourceInstantiationException;
import gate.creole.SerialAnalyserController;
import gate.persist.PersistenceException;
import gate.util.GateException;
import gate.util.persistence.PersistenceManager;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.MalformedURLException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * Example code running a GATE pipeline
 *
 * @author fredrik.axelsson
 */
public class Example {

    private static final Logger log = LoggerFactory.getLogger(Example.class);

    private SerialAnalyserController pipeline;

    public static void main(String[] args) throws GateException, PersistenceException, IOException, URISyntaxException {
        
        if (args.length < 3) {
            System.out.println("Please, provide a directory containing a pipeline and a path to process");
            System.out.println("java -jar GateEmbeddedExample-1.0-SNAPSHOT.jar <pipeline.dir> <document path> <outputdir>");
            System.exit(0);
        }
        
        Path gatehome = Paths.get(args[0]);        
        File pipeline = gatehome.resolve("application.xgapp").toFile();
        

        Example example = new Example(gatehome.toFile(), pipeline);
        Path pathToDoc = Paths.get(args[1]);
        String output = example.processPath(pathToDoc);
        
        Path outFile = Paths.get(args[2]);
        try (PrintWriter pw = new PrintWriter(outFile.toFile())) {
            pw.write(output);
            pw.flush();
        }
    }

    public Example(File gateHome, File pipeline) throws GateException, PersistenceException, IOException {
        initGate(gateHome);
        initPipeline(pipeline);
    }

    private void initGate(File gateHome) throws GateException {
        Gate.setGateHome(gateHome);
        Gate.init();
    }

    private void initPipeline(File pipelineFile) throws PersistenceException, IOException, ResourceInstantiationException {
        pipeline = (SerialAnalyserController) PersistenceManager.loadObjectFromFile(pipelineFile);
    }
    
    public String processString(String text) throws ResourceInstantiationException, ExecutionException {
        log.info("Start processing");
        Document document = Factory.newDocument(text);
        Corpus corpus = Factory.newCorpus("test corpus");
        corpus.add(document);
        pipeline.setCorpus(corpus);
        pipeline.execute();
        log.info("Processing done");

        return document.toXml();
    }
    
    public String processPath(Path path) throws MalformedURLException, ResourceInstantiationException, ExecutionException {
        log.info("Start processing");
        URL sourceUrl = path.toUri().toURL();
        Document document = Factory.newDocument(sourceUrl);
        Corpus corpus = Factory.newCorpus("test corpus");
        corpus.add(document);
        pipeline.setCorpus(corpus);
        pipeline.execute();
        log.info("Processing done");

        return document.toXml();
    }

}
