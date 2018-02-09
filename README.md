# Neural

Install Java 9 JDK and set to path in environment variables. Install latest maven and set to path. 
You can use latest eclipse with maven project capability as an IDE. 

To run maven install in console
mvn clean install

To run prediction run in console: 
mvn exec:java -Dexec.args="2"

To train the network run:
mvn exec:java -Dexec.args="1"
