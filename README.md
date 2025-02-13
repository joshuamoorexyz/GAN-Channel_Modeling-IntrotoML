### LTSM-GAN Channel Modeling for UAV-to-Ground CommunicationS

---

### **Project Title**
**Enhancing UAV-to-Ground Channel Modeling for AERPAW Digital Twin Using Generative Adversarial Networks**

---

### **Problem Statement**

Accurate wireless channel measurements are essential for
the AERPAW digital twin to realistically simulate UAV-
to-ground communication in dynamic, mobility-driven en-
vironments. However, collecting real-world channel mea-
surements is often time-consuming, resource-intensive, and
challenging due to the need for extensive field testing under
varying environmental conditions. While statistical models,
such as Two-Ray Ground Reflection models, can simulate
channel behavior, they lack the flexibility to capture the
complex, environment-specific conditions of UAV operations.
These models often fail to reflect the variability introduced
by changing altitudes, rapid Doppler shifts, and intricate mul-
tipath propagation effects, limiting their ability to represent
the true dynamics of UAV communication channels.

---

### **Proposed Solution**

To address these limitations, we will employ a Generative Adversarial Network (GAN) to create high-quality synthetic channel measurements. By integrating a Long Short-Term Memory (LSTM) network within the GAN architecture, we can effectively model the temporal dependencies present in our collected time-series UAV-to-ground communication data. Leveraging metrics such as Signal-to-Noise Ratio (SNR), path loss, and the angles of arrival and departure, the GAN will learn patterns unique to UAV channels, producing realistic and context-aware synthetic data. This approach will enhance the fidelity of the digital twin, offering more granular insights into UAV-specific communication scenarios .


<img src="/diagram.png">


---

### **Team Members**

| Name                | netID   | GitHub                                                                                  |
|---------------------|---------|-----------------------------------------------------------------------------------------|
| **Joshua Moore**    | jjm702  | [![GitHub](https://skillicons.dev/icons?i=github)](https://github.com/joshuamoorexyz)   |
| **Tirian Judy**     | tkj105  | [![GitHub](https://skillicons.dev/icons?i=github)](https://github.com/Tirian33)        |
| **Aayam Raj Shakya**| as5160  | [![GitHub](https://skillicons.dev/icons?i=github)](https://github.com/aayamrajshakya)  |
| **Claire Johnson**  | kj1289  | [![GitHub](https://skillicons.dev/icons?i=github)](https://github.com/clairejohnson0714)  |
